from PLAP.llm_agents.llm_agent import LLM, Qwen, GPT, DeepSeek, CLAUDE, GEMINI
from copy import deepcopy

from typing import Tuple, List


def CONST_QUESTIONS() -> Tuple[list, list]:
    urts_questions = [
        {
            "question": "How many time units does it take to build the Base in MicroRTS?",
            "answer": "250",
        },
        {"question": "How many hit points of the Barrack in MicroRTS?", "answer": "4"},
        {
            "question": "How many resources does it cost to build the Light in MicroRTS?",
            "answer": "2",
        },
        {
            "question": "How many time units does it take to build the Light in MicroRTS?",
            "answer": "80",
        },
        {
            "question": "How many damage does the attack of the Ranged in MicroRTS?",
            "answer": "1",
        },
    ]
    sc2_questions = [
        {
            "question": "How many hit points of Terran SCV in StarCraft II (SC2)?",
            "answer": "45",
        },
        {
            "question": "How many hit points of Zerg Viper in StarCraft II (SC2)?",
            "answer": "150",
        },
        {
            "question": "How many hit points of Terran Thor in StarCraft II (SC2)?",
            "answer": "400",
        },
        {
            "question": "How many transport slots of Medivac dropship in StarCraft II (SC2)?",
            "answer": "8",
        },
        {
            "question": "How many minerals does it cost to produce a Stalker of Protoss in StarCraft II (SC2)?",
            "answer": "125",
        },
    ]
    return urts_questions, sc2_questions


def llm_question(llm: LLM, questions: List[dict], instruction="\n"):
    summary = deepcopy(questions)

    for i in range(len(questions)):
        prompt = questions[i]["question"] + instruction
        res = llm(prompt)
        summary[i]["response"] = res

        try:
            result = res[res.find("[") + 1 : res.find("]")]
        except:
            summary[i]["result"] = "NONE"
            continue
        summary[i]["result"] = result

    info, cr = correct_rate(summary)
    print(f"QA Details: \n{summary}")
    print(f"Correct Rate: {cr}:{info}")
    return summary, cr


def correct_rate(summary: List[dict]) -> float:
    info = [1 if q["answer"] == q["result"] else 0 for q in summary]
    return info, sum(info) / len(summary)


if __name__ == "__main__":
    q_urts, q_sc2 = CONST_QUESTIONS()
    ins = "\nPlease finally include the answer in square brackets, e.g. [1234].\n"
    # ins = '\n'

    qwen = Qwen("Qwen2-72B-Instruct", 0, 1024)
    deep = DeepSeek("deepseek-chat", 0, 1024)
    claude3hai = CLAUDE("claude-3-haiku-20240307", 0, 1024)
    claude35son = CLAUDE("claude-3-5-sonnet-20240620", 0, 1024)
    gemini15flash = GEMINI("gemini-1.5-flash-002", 0, 1024)
    gpt35 = GPT("gpt-3.5-turbo", 0, 1024)
    gpt4m = GPT("gpt-4o-mini", 0, 1024)
    gpt4o = GPT("gpt-4o", 0, 1024)

    skill_rate = []
    sc2_rate = []

    print(f"{'*'*10}Qwen2-72B: MicroRTS QA{'*'*10}")
    summary, rate = llm_question(qwen, q_urts, instruction=ins)
    skill_rate.append(rate)
    print(f"{'*'*10}Qwen2-72B: StarCraftII QA{'*'*10}")
    summary, rate = llm_question(qwen, q_sc2, instruction=ins)
    sc2_rate.append(rate)

    print(f"{'*'*10}deepseek-chat: MicroRTS QA{'*'*10}")
    summary, rate = llm_question(deep, q_urts, instruction=ins)
    skill_rate.append(rate)
    print(f"{'*'*10}deepseek-chat: StarCraftII QA{'*'*10}")
    summary, rate = llm_question(deep, q_sc2, instruction=ins)
    sc2_rate.append(rate)

    print(f"{'*'*10}claude-3-haiku: MicroRTS QA{'*'*10}")
    summary, rate = llm_question(claude3hai, q_urts, instruction=ins)
    skill_rate.append(rate)
    print(f"{'*'*10}claude-3-haiku: StarCraftII QA{'*'*10}")
    summary, rate = llm_question(claude3hai, q_sc2, instruction=ins)
    sc2_rate.append(rate)

    print(f"{'*'*10}claude-3-5-sonnet: MicroRTS QA{'*'*10}")
    summary, rate = llm_question(claude35son, q_urts, instruction=ins)
    skill_rate.append(rate)
    print(f"{'*'*10}claude-3-5-sonnet: StarCraftII QA{'*'*10}")
    summary, rate = llm_question(claude35son, q_sc2, instruction=ins)
    sc2_rate.append(rate)

    print(f"{'*'*10}gemini-1.5-flash: MicroRTS QA{'*'*10}")
    summary, rate = llm_question(gemini15flash, q_urts, instruction=ins)
    skill_rate.append(rate)
    print(f"{'*'*10}gemini-1.5-flash: StarCraftII QA{'*'*10}")
    summary, rate = llm_question(gemini15flash, q_sc2, instruction=ins)
    sc2_rate.append(rate)

    print(f"{'*'*10}gpt-3.5-turbo: MicroRTS QA{'*'*10}")
    summary, rate = llm_question(gpt35, q_urts, instruction=ins)
    skill_rate.append(rate)
    print(f"{'*'*10}gpt-3.5-turbo: StarCraftII QA{'*'*10}")
    summary, rate = llm_question(gpt35, q_sc2, instruction=ins)
    sc2_rate.append(rate)

    print(f"{'*'*10}gpt-4o-mini: MicroRTS QA{'*'*10}")
    summary, rate = llm_question(gpt4m, q_urts, instruction=ins)
    skill_rate.append(rate)
    print(f"{'*'*10}gpt-4o-mini: StarCraftII QA{'*'*10}")
    summary, rate = llm_question(gpt4m, q_sc2, instruction=ins)
    sc2_rate.append(rate)

    print(f"{'*'*10}gpt-4o: MicroRTS QA{'*'*10}")
    summary, rate = llm_question(gpt4o, q_urts, instruction=ins)
    skill_rate.append(rate)
    print(f"{'*'*10}gpt-4o: StarCraftII QA{'*'*10}")
    summary, rate = llm_question(gpt4o, q_sc2, instruction=ins)
    sc2_rate.append(rate)

    print(f"{'*'*10}correct rates: MicroRTS QA{'*'*10}")
    print(skill_rate)
    print(f"{'*'*10}correct rates: StarCraftII QA{'*'*10}")
    print(sc2_rate)
