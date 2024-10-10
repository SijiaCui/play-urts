import yaml
import tiktoken
from openai import OpenAI

from PLAR.llm_agents.prompts import zero_shot_prompt, few_shot_prompt, reflect_prompt, zero_shot_w_tips, few_shot_w_tips
from PLAR.utils.utils import parse_tips
import PLAR.utils as utils

class LLMAgent:

    def __init__(self, engine, temperature, max_tokens, map_name, prompt_config) -> None:
        if "qwen" in engine.lower():
            self.llm = Qwen(engine, temperature, max_tokens)
        elif "gpt" in engine.lower() or "o1" in engine.lower():
            self.llm = GPT(engine, temperature, max_tokens)
        elif "llama" in engine.lower():
            self.llm = Llama(engine, temperature, max_tokens)
        elif "deepseek" in engine.lower():
            self.llm = DeepSeek(engine, temperature, max_tokens)
        else:
            raise ValueError("Invalid engine name")
        self.map_name = map_name
        self.prompt_config = prompt_config

    def _post_processing(self, response: str) -> str:
        return response

    def _agent_prompt(self) -> str:
        with open(f"/root/desc/play-urts/PLAR/configs/prompts/{self.map_name}_few_shot.yaml") as f:
            content = yaml.safe_load(f)
        examples = content["EXAMPLES"]
        tips = content["TIPS"]
        if self.prompt_config == "zero_shot":
            self.prompt = zero_shot_prompt
            kwargs = {
                "observation": self.obs,
                "fight_for": utils.FIGHT_FOR,
            }
        elif self.prompt_config == "few_shot":
            self.prompt = few_shot_prompt
            kwargs = {
                "examples": examples,
                "observation": self.obs,
                "fight_for": utils.FIGHT_FOR,
            }
        elif self.prompt_config == "zero_shot_w_tips":
            self.prompt = zero_shot_w_tips
            kwargs = {
                "tips": tips,
                "observation": self.obs,
                "fight_for": utils.FIGHT_FOR,
            }
        elif self.prompt_config == "few_shot_w_reflect":
            tips = parse_tips(self.reflect())
            self.prompt = few_shot_w_tips
            kwargs = {
                "examples": examples,
                "tips": tips,
                "observation": self.obs,
                "fight_for": utils.FIGHT_FOR,
            }
        elif self.prompt_config == "few_shot_w_tips":
            self.prompt = few_shot_w_tips
            kwargs = {
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
        with open(f"/root/desc/play-urts/PLAR/configs/prompts/{self.map_name}_few_shot.yaml") as f:
            examples = yaml.safe_load(f)["TIPS"]
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


class GPT(LLM):
    def __init__(self, engine, temperature, max_tokens) -> None:
        import os

        self.client = OpenAI(base_url="https://api.openai-proxy.live/v1", api_key=os.getenv("OPENAI_API_KEY"))
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
        token_limit_gpt = 128000
        return len(tiktoken.encoding_for_model(self.engine).encode(prompt)) > token_limit_gpt


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


class DeepSeek(LLM):
    def __init__(self, engine, temperature, max_tokens) -> None:
        from openai import OpenAI
        import os
        self.client = OpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))

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
            stream=False
        )
        return response.choices[0].message.content
    
    def is_excessive_token(self, prompt) -> bool:
        token_limit_deepseek = 128000
        return len(prompt) > token_limit_deepseek


if __name__ == "__main__":
    # llm_agent = LLMAgent("gpt-4o-mini", 0, 1024, None, None)
    # llm_agent = LLMAgent("deepseek-chat", 0, 1024, None, None)
    llm_agent = LLMAgent("Qwen2-72B-Instruct", 0, 1024, None, None)
    print(llm_agent.llm("who are you"))
