COA_MANUAL = """
"""

COA_OPPONENT = """
"""

COA_EXAMPLES = """The Situation: There are 2 mineral fields in this map. The mineral field located in (0, 0) has 4 available resources. The mineral field located in (3, 3) has 4 available resources. The red team has one base located in (2, 1) with 0 remaining resource, 4 remaining HP, and the current action of it is noop. The red team has no barrack. The red team has one worker located in (2, 2), which carries 0 resource and the current action is noop. The red team has no light soldier. The red team has no heavy soldier. The red team has no ranged soldier. The blue team has one base located in (1, 2) with 0 remaining resource, 4 remaining HP, and the current action of it is noop. The blue team has no barrack. The blue team has one worker located in (1, 1), which carries 0 resource and the current action is noop. The blue team has no light soldier. The blue team has no heavy soldier. The blue team has no ranged soldier. 
Course of Action: 
START of COA
1. [Attack Enemy Buildings]
2. [Produce Worker]
3. ...
END of COA
"""


COA_DETAILED_EXAMPLES = """
"""

COA_H_Mineral = "[Harvest Mineral]"
COA_B_Base = "[Build Base]"
COA_B_Barrack = "[Build Barrack]"

COA_P_Worker = "[Produce Worker]"
COA_P_Light = "[Produce Light Soldier]"
COA_P_Heavy = "[Produce Heavy Soldier]"
COA_P_Ranged = "[Produce Ranged Soldier]"

COA_A_Worker = "[Attack Enemy Worker]"
COA_A_Buildings = "[Attack Enemy Buildings]"
COA_A_Soldiers = "[Attack Enemy Soldiers]"

COA_ACTION_SPACE = [
    COA_H_Mineral,
    COA_B_Base,
    COA_B_Barrack,
    COA_P_Worker,
    COA_P_Light,
    COA_P_Heavy,
    COA_P_Ranged,
    COA_A_Worker,
    COA_A_Buildings,
    COA_A_Soldiers,
]
COA_ACTION_SPACE_STR = f"{{{', '.join(COA_ACTION_SPACE)}}}"
