import os
import tiktoken

from langchain_openai import AzureChatOpenAI

from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


class coa_agent:
    def __init__(self, args) -> None:
        self.args = args

        # qwen: qwen, gpt: azuregpt, llama: llama2
        self.llm = qwen(
            self.args.llm_engine, 
            self.args.llm_temperature, 
            self.args.llm_max_tokens
            )

        self.step = 0

        pass

    def _agent_prompt(self) -> str:
        kwargs = {
            
        }

        self.prompt.format(**kwargs)


    def _agent_response(self) -> str:
        prompt = self._agent_prompt()


        pass


    def run(self) -> str:
        
        
        self.llm()
    


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
        super().__call__(self, prompt)
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
        super().__call__(self, prompt)
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

    ca = coa_agent()
