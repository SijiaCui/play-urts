from langchain.prompts import PromptTemplate

# ========== START: tdqa提示模板 ==========

# 问题分类器的prompt
CLASSIFIER_INSTRUCTION = """
You're an expert at question classification. 
Given a question that is sampled from 8 distinct datasets, you can extract the keywords from the question and figure out the most possible dataset that the question belongs to. 
You should think step by step and output the final classification result [DATASET].

Here are descriptions of all 8 datasets:
{databases_info}(END OF DESCRIPTIONS)

Note that your answer must include a classification result in the form of [DATASET], where DATASET is your classification result.
Please use the following format when answering:
Question: This is the question.
Answer: These are your thoughts. [DATASET]
Here are some examples:
{examples_classifier}(END OF EXAMPLES)
Question: {question}"""

CLASSIFIER_INSTRUCTION_NO_INFO = """
You're an expert at question classification. 
Given a question that is sampled from 8 distinct datasets, you can extract the keywords from the question and figure out the most possible dataset that the question belongs to. 
You should think step by step and output the final classification result [DATASET].
Note that your answer must include a classification result in the form of [DATASET], where DATASET is your classification result.
Please use the following format when answering:
Question: This is the question.
Answer: These are your thoughts. 
[DATASET]
Here are some examples:
{examples_classifier}(END OF EXAMPLES)
Question: {question}"""

CLASSIFIER_INSTRUCTION_ZERO_SHOT = """
You're an expert at question classification. 
Given a question that is sampled from 8 distinct datasets, you can extract the keywords from the question and figure out the most possible dataset that the question belongs to. 
You should think step by step and output the final classification result [DATASET].
Here are descriptions of all 8 datasets:
{databases_info}(END OF DESCRIPTIONS)
Note that your answer must include a classification result in the form of [DATASET], where DATASET is your classification result.
Please use the following format when answering:
Question: This is the question.
Answer: These are your thoughts. [DATASET]
Question: {question}"""

classifier_prompt = PromptTemplate(
    input_variables=["databases_info", "examples_classifier", "question"],
    template=CLASSIFIER_INSTRUCTION,
)
classifier_prompt_no_info = PromptTemplate(
    input_variables=["examples_classifier", "question"],
    template=CLASSIFIER_INSTRUCTION_NO_INFO,
)
classifier_prompt_zero_shot = PromptTemplate(
    input_variables=["databases_info", "question"],
    template=CLASSIFIER_INSTRUCTION_ZERO_SHOT,
)


# 任务分解器prompt
DECOMPOSER_INSTRUCTION = """You are an expert in task decomposition. 
Given a complex question and a set of tools, your mission is to generate the next tool to be called and the corresponding subtask based on historical information. 
Your output should be in the form of [TOOL]<SUBTASK>, where [TOOL] is the next tool to complete the subtask <SUBTASK>.
Here are descriptions of all available tools:
{tools_info}(END OF DESCRIPTIONS)
Please use the following format when answering(n is a natural number):
Question: The question.
Hint: The hint to answer the question.
Subtask 1: Your thoughts. 
[TOOL]<SUBTASK>
Result 1: The execution result of Subtask 1.
...
Subtask n: Your thoughts. 
[Finish]<SUBTASK>
Result n: The execution result of Subtask n.
Final Answer: The final answer.
Here are some examples:
{examples_decomposer}(END OF EXAMPLES)
Question: {question}
{scratchpad}"""

decomposer_prompt = PromptTemplate(
    input_variables=["tools_info", "examples_decomposer", "question", "scratchpad"],
    template=DECOMPOSER_INSTRUCTION,
)

# 子任务处理器prompt：tool-level prompt
from prompts_tools import TOOLS_PROMPT_LIST
from prompts_tools import ONE_DIRECT_PROMPT

# ========== END: tdqa提示模板 ==========


# 子任务处理器prompt：1.直接生成参数
HANDLER_INSTRUCTION_DIRECT = """Solve a question-answering task by using tools. 
Task Decomposer and Subtask Handler must work together to complete the task. 
Now, you are a Subtask Handler, your mission is to generate parameters of [TOOL] for completing the <SUBTASK> decomposed by the decomposer and your output should be in the form of [TOOL]<PARAMETERS>. 
The <PARAMETERS> denotes the needed parameters for [TOOL] to complete the <SUBTASK> correctly.
Here are some parameter descriptions about [TOOL]:
{parameter_info}
(END OF PARAMETER DESCRIPTIONS)
Here are some examples:
{examples_decomposer_handler}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""

# 子任务处理器prompt：2.参数组合
HANDLER_INSTRUCTION_COMBO = """
"""

# 子任务处理器prompt：3.分层分解
HANDLER_INSTRUCTION_HIERARCHICAL = """
"""

handler_prompt_direct = PromptTemplate(
    input_variables=[
        "parameter_info",
        "examples_decomposer_handler",
        "question",
        "scratchpad",
    ],
    template=HANDLER_INSTRUCTION_DIRECT,
)
handler_prompt_combo = PromptTemplate(
    input_variables=["examples", "question", "scratchpad"],
    template=HANDLER_INSTRUCTION_COMBO,
)
handler_prompt_hierarchical = PromptTemplate(
    input_variables=["examples", "question", "scratchpad"],
    template=HANDLER_INSTRUCTION_HIERARCHICAL,
)

VERIFIER_INSTRUCTION = """
"""

verifier_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad"],
    template=VERIFIER_INSTRUCTION,
)

SUMMARIZER_INSTRUCTION = """
"""

summarizer_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad"],
    template=SUMMARIZER_INSTRUCTION,
)

# ========== react方法中的提示模板 ==========
COT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. 
Thought can reason about the current situation. 
Finish[answer] returns the answer and finishes the task. 
You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Relevant Context: {context} 
Question: {question}{scratchpad}"""

COT_AGENT_REFLECT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. 
Thought can reason about the current situation. 
Finish[answer] returns the answer and finishes the task. 
You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Relevant Context: {context}
Question: {question}{scratchpad}"""

COT_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. 
You will be given a previous reasoning trial in which you were given access to relevant context and a question to answer. 
You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. 
In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. 
Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Relevant Context: {context}
Question: {question}{scratchpad}

Reflection:"""

cot_agent_prompt = PromptTemplate(
    input_variables=["examples", "reflections", "context", "question", "scratchpad"],
    template=COT_INSTRUCTION,
)

cot_reflect_agent_prompt = PromptTemplate(
    input_variables=["examples", "reflections", "context", "question", "scratchpad"],
    template=COT_AGENT_REFLECT_INSTRUCTION,
)

cot_reflect_prompt = PromptTemplate(
    input_variables=["examples", "context", "question", "scratchpad"],
    template=COT_REFLECT_INSTRUCTION,
)

COT_SIMPLE_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. 
Thought can reason about the current situation. 
Finish[answer] returns the answer and finishes the task.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
{context}
Question: {question}{scratchpad}"""

COT_SIMPLE_AGENT_REFLECT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. 
Thought can reason about the current situation. 
Finish[answer] returns the answer and finishes the task.
Here are some examples:
{examples}
(END OF EXAMPLES)
{context}
{reflections}

Question: {question}{scratchpad}"""

COT_SIMPLE_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. 
You will be given a previous reasoning trial in which you were given a question to answer. 
You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. 
In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. 
Use complete sentences.
Here are some examples:
{examples}
(END OF EXAMPLES)
{context}
Previous trial:
Question: {question}{scratchpad}

Reflection:"""

cot_simple_agent_prompt = PromptTemplate(
    input_variables=["examples", "question", "reflections", "context", "scratchpad"],
    template=COT_SIMPLE_INSTRUCTION,
)

cot_simple_reflect_agent_prompt = PromptTemplate(
    input_variables=["examples", "context", "reflections", "question", "scratchpad"],
    template=COT_SIMPLE_AGENT_REFLECT_INSTRUCTION,
)

cot_simple_reflect_prompt = PromptTemplate(
    input_variables=["examples", "question", "context", "scratchpad"],
    template=COT_SIMPLE_REFLECT_INSTRUCTION,
)


REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be 13 types: 
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda related to keyword.
(3) RetrieveScirex[keyword], which retrieves machine learning papers' paragraphs related to keyword.
(4) LoadDB[DBName], which loads the database DBName and returns the database. 
The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. 
The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""

REACT_REFLECT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be 13 types: 
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda containing keyword and returns the agenda.
(3) RetrieveScirex[keyword], which retrieves the most relevant paragraphs in machine learning-related papers to keyword and returns the paragraphs.
(4) LoadDB[DBName], which loads the database DBName and returns the database. 
The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. 
The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""

REFLECTION_HEADER = """
You have attempted to answer following question before and failed. 
The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. 
Use them to improve your strategy of correctly answering the given question.\n"
REFLECTION_AFTER_LAST_TRIAL_HEADER = "The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. 
Use them to improve your strategy of correctly answering the given question.\n"
LAST_TRIAL_HEADER = "You have attempted to answer the following question before and failed. 
Below is the last trial you attempted to answer the question.\n
"""

REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. 
You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. 
You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. 
In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. 
Use complete sentences.  
Here are some examples:
{examples}

Previous trial:
Question: {question}{scratchpad}

Reflection:"""

react_agent_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad"],
    template=REACT_INSTRUCTION,
)

react_reflect_agent_prompt = PromptTemplate(
    input_variables=["examples", "reflections", "question", "scratchpad"],
    template=REACT_REFLECT_INSTRUCTION,
)

reflect_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad"],
    template=REFLECT_INSTRUCTION,
)
