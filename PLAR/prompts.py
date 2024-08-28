from langchain.prompts import PromptTemplate

COA_INSTRUCTION = """You are an expert at generating course of action based on observation. 
{observation}
You should 
"""

coa_prompt = PromptTemplate(
    input_variables = ["observation", ""],
    template = COA_INSTRUCTION,
)
