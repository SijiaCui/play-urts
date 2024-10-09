import tiktoken
import yaml
from langchain_openai import AzureChatOpenAI

from langchain_core.messages import HumanMessage

from PLAR.llm_agents.prompts import zero_shot_prompt, few_shot_prompt, prompt_w_tips, prompt_w_opponent, reflect_prompt
from PLAR.utils.utils import parse_tips
import PLAR.utils as utils

class LLMAgent:

    def __init__(self, engine, temperature, max_tokens, map_name, prompt_config) -> None:
        # qwen: qwen, gpt: azuregpt, llama: llama2
        if "qwen" in engine.lower():
            self.llm = Qwen(engine, temperature, max_tokens)
        elif "gpt" in engine.lower():
            self.llm = AzureChatOpenAI(engine, temperature, max_tokens)
        elif "llama" in engine.lower():
            self.llm = Llama(engine, temperature, max_tokens)
        else:
            raise ValueError("Invalid engine name")
        self.map_name = map_name
        self.prompt_config = prompt_config

    def _post_processing(self, response: str) -> str:
        return response

    def _agent_prompt(self) -> str:
        with open(f"/root/desc/play-urts/PLAR/configs/templates/{self.prompt_config[0]}/instruction.txt", "r") as f:
            instruction = f.read()
        with open(f"/root/desc/play-urts/PLAR/configs/templates/{self.prompt_config[0]}/{self.map_name}_few_shot.yaml") as f:
            examples = yaml.safe_load(f)["EXAMPLES"]
        if self.prompt_config[1] == "zero_shot_prompt":
            self.prompt = zero_shot_prompt
            kwargs = {
                "instruction": instruction,
                "observation": self.obs,
                "fight_for": utils.FIGHT_FOR,
            }
        elif self.prompt_config[1] == "few_shot_prompt":
            self.prompt = few_shot_prompt
            kwargs = {
                "instruction": instruction,
                "examples": examples,
                "observation": self.obs,
                "fight_for": utils.FIGHT_FOR,
            }
        elif self.prompt_config[1] == "prompt_w_reflect_tips":
            tips = parse_tips(self.reflect())
            self.prompt = prompt_w_tips
            kwargs = {
                "instruction": instruction,
                "examples": examples,
                "tips": tips,
                "observation": self.obs,
                "fight_for": utils.FIGHT_FOR,
            }
        elif self.prompt_config[1] == "prompt_w_expert_tips":
            with open(f"/root/desc/play-urts/PLAR/configs/templates/reflection/{self.map_name}_few_shot.yaml") as f:
                tips = yaml.safe_load(f)["EXAMPLES"]
            self.prompt = prompt_w_tips
            kwargs = {
                "instruction": instruction,
                "examples": examples,
                "tips": tips,
                "observation": self.obs,
                "fight_for": utils.FIGHT_FOR,
            }
        return self.prompt.format(**kwargs)

    def _agent_response(self) -> str:
        prompt_content = self._agent_prompt()
        try:
            response = self.llm(prompt=prompt_content)
        except Exception as e:
            response = f"LLM response error: {e}"
        print(prompt_content)
        print(response)
        return response

    def reflect(self):
        with open(f"/root/desc/play-urts/PLAR/configs/templates/reflection/{self.map_name}_few_shot.yaml") as f:
            examples = yaml.safe_load(f)["EXAMPLES"]
        prompt_content = reflect_prompt.format(
            examples=examples,
            observation=self.obs,
            fight_for=utils.FIGHT_FOR
        )
        try:
            response = self.llm(prompt=prompt_content)
        except Exception as e:
            response = f"LLM response error: {e}"
        return response

    def run(self, obs) -> str:
        self.obs = obs
        self.response = self._agent_response()
        return self.response


class LLM:
    def __init__(self) -> None:
        pass

    def __call__(self, prompt) -> str:
        if self.is_excessive_token(prompt):
            raise ValueError("Error: prompt is too long.")

    def is_excessive_token(self, prompt) -> bool:
        pass


class AzureGPT(LLM):
    def __init__(self, engine, temperature, max_tokens) -> None:
        self.client = AzureChatOpenAI(
            deployment_name = engine,
            temperature = temperature, 
            max_tokens = max_tokens
        )
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, prompt) -> str:
        super().__call__(prompt)
        self.client([HumanMessage(content=prompt)]).content

    def is_excessive_token(self, prompt) -> bool:
        token_limit_gpt35 = 3896 # gpt3.5 4k, set limit 3896
        return tiktoken.encoding_for_model(self.engine).encode(prompt) > token_limit_gpt35


class Qwen(LLM):
    def __init__(self, engine, temperature, max_tokens) -> None:
        from openai import OpenAI
        import os
        if engine == 'Qwen1.5-72B-Chat':
            self.client = OpenAI(base_url=os.getenv("QWEN_BASE"), api_key="XXX")
        elif engine == 'Qwen2-72B-Instruct':
            self.client = OpenAI(base_url=os.getenv("QWEN2_BASE"), api_key=os.getenv("QWEN2_KEY"))
        else:
            raise ValueError("Error: wrong engine name")
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, prompt) -> str:
        super().__call__(prompt)
        response = self.client.chat.completions.create(
            model=self.engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
    
    def is_excessive_token(self, prompt) -> bool:
        token_limit_qwen32k = 32*1024 # set limit <str length> 32k (approximately 8k token)
        return len(prompt) > token_limit_qwen32k


class Llama(LLM):
    pass
