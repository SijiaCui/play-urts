from langchain.prompts import PromptTemplate

COA_INSTRUCTION = """You are an expert at answering question. 
The question is : {observation}
Your Answer is: 
"""

coa_prompt = PromptTemplate(
    input_variables = ["observation", "examples"],
    template = COA_INSTRUCTION,
)
