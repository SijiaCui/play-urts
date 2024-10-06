import yaml
from langchain.prompts import PromptTemplate


with open("/root/desc/play-urts/PLAR/configs/templates/base_template.yaml") as f:
    base_template = yaml.safe_load(f)

INSTRUCTION = base_template["INSTRUCTION"]
INTRODUCTION = base_template["INTRODUCTION"]
EXAMPLES = base_template["EXAMPLES"]
TIPS = base_template["TIPS"]
OPPONENT = base_template["OPPONENT"]
START = base_template["START"]

zero_shot_prompt = PromptTemplate(
    input_variables=["instruction", "observation", "fight_for"],
    template=INSTRUCTION + INTRODUCTION + START,
)

few_shot_prompt = PromptTemplate(
    input_variables=["instruction", "examples", "observation", "fight_for"],
    template=INSTRUCTION + INTRODUCTION + EXAMPLES + START,
)

prompt_w_tips = PromptTemplate(
    input_variables=["instruction", "examples", "tips", "observation", "fight_for"],
    template=INSTRUCTION + INTRODUCTION + EXAMPLES + START + TIPS,
)

prompt_w_opponent = PromptTemplate(
    input_variables=["instruction", "examples", "opponent", "observation", "fight_for"],
    template=INSTRUCTION + INTRODUCTION + EXAMPLES + OPPONENT + START,
)

with open("/root/desc/play-urts/PLAR/configs/templates/reflect_template.yaml") as f:
    reflect_template = yaml.safe_load(f)

REFLECT_TEMPLATE = reflect_template["REFLECT_TEMPLATE"]
reflect_prompt = PromptTemplate(
    input_variables=["examples", "observation", "fight_for"],
    template=REFLECT_TEMPLATE
)
