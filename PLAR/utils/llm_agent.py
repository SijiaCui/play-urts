import tiktoken

from langchain_openai import AzureChatOpenAI

from langchain_core.messages import HumanMessage

from PLAR.utils.prompts import zero_shot_prompt, few_shot_prompt, prompt_w_tip, prompt_w_opponent, MANUAL, OPPONENT, TASK_SPACE, EXAMPLES, TIP

class LLMAgent:
    def __init__(self, args) -> None:
        self.args = args

        # qwen: qwen, gpt: azuregpt, llama: llama2
        self.llm = qwen(
            self.args.engine, 
            self.args.temperature, 
            self.args.max_tokens
            )

        self.step = 0
        self.prompt = prompt_w_tip

    def _post_processing(self, response: str) -> str:

        return response

    def _agent_prompt(self) -> str:
        kwargs = {
            "manual": MANUAL,
            "task_space": TASK_SPACE,
            "examples": EXAMPLES,
            "observation": self.obs,
            "tip": TIP,
        }

        return self.prompt.format(**kwargs)

    def _agent_response(self) -> str:
        LLM_DEBUG = self.args.debug
        if LLM_DEBUG: print(f"{'-'*20}#{self.__class__.__name__} START DEBUGGING#{'-'*20}", flush=True)
        prompt_content = self._agent_prompt()
        print(prompt_content)
        if LLM_DEBUG: print(f"PROMPT: {self.prompt.input_variables}\n{prompt_content}", flush=True)

        try:
            response = self.llm(prompt=prompt_content)
        except Exception as e:
            response = f"llm reponse error: {e}"

        if LLM_DEBUG: print(f"RESPONSE: {response}", flush=True)
        response = self._post_processing(response)
        if LLM_DEBUG: print(f"RESPONSE POST_PROCESSING: {response}", flush=True)
        if LLM_DEBUG: print(f"{'-'*20}#{self.__class__.__name__} END DEBUGGING#{'-'*20}", flush=True)

        return response

    def run(self, obs) -> str:

        self.obs = obs

        self.response = self._agent_response()

        return self.response


class llms:
    def __init__(self) -> None:
        pass
    def __call__(self, prompt) -> str:
        if self.is_excessive_token(prompt):
            raise ValueError("Error: prompt is too long.")
    def is_excessive_token(self, prompt) -> bool:
        pass


class azuregpt(llms):
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


class qwen(llms):
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


class llama2(llms):
    pass


if __name__ == "__main__":
    from PLAR.utils.utils import load_args
    args = load_args()

    obs = """Available Mineral Fields: 2
- Mineral Fields(0, 0) resource: 16
- Mineral Fields(7, 7) resource: 13

Red's Units:
base: 1
- (6, 5), action: noop
barrack: 0
worker: 3
- (6, 4), action: noop
- (6, 7), action: noop
- (7, 5), action: noop
light: 0
heavy: 0
ranged: 0

Blue's Units:
base: 1
- (1, 2), task: [noop], action: noop
barrack: 0
worker: 3
- (0, 2), task: [noop], action: noop
- (1, 1), task: [noop], action: noop
- (1, 3), task: [noop], action: move
light: 0
heavy: 0
ranged: 0
"""

    llm_agent = LLMAgent(args)
    response = llm_agent.run(obs)

    print(response)
