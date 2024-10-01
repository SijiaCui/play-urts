# TODO
- [x] ranged, heavy, light 不同是什么
- [x] task to actions
- [x] task assignment
- [x] some bugs
  - [x] ranged auto attack
- [ ] call llm
  - [ ] rewrite prompt
    - 任务和参数要求
    - 游戏规则
    - 示例
  - [ ] Exp.
    - [ ] low-level: CoT/ReAct/LLM
    - [ ] high-level: CoT/ReAct/LLM
      - [ ] Expert Rules!!!
    - [ ] Ours: CoT/ReAct/LLM


## LLM PLAY RTS

- LLM output task format, `[task](params)`
    - `[Deploy Unit](unit_type, tgt_loc)`   # 持续性任务
    - `[Harvest Mineral](mineral_loc, tgt_loc, base_loc, return_loc)`  # 持续任务
    - `[Build Building](building_type, building_loc, tgt_loc)`  # 一次性任务
    - `[Produce Unit](produce_type, direction)`  # 一次性任务
    - `[Attack Enemy](unit_type, enemy_type, enemy_loc, tgt_loc)`  # 一次性任务

## Baselines

### Rule

...

### LLM ACT with low-level action

LLM/CoT/ReAct

# 问题

- 正在建造的 object 挡路
- units 都没有 ID，不知道谁是谁
- ！！！基地资源 obs 一直是 0（环境问题）

# 1003
## SijiaCui
[] Paper: Abstract
[] Paper: Method
[] highlight the point in the figure -- skill lib

## ShuaiXu
[] Exp prompt: {manual}, {examples}, [opponent]
[] Exp Baseline: low-level: CoT/ReAct(examples)

## AiyaoHe
[] Paper: Related Work

## Others
    High-level: TextSC2 adaptation, -> XXX --expertRule--> Parameter, 
    Harvest Mineral -> expertRule --> 1,2,3,4
    Oracel
    Harvest Mineral(1,2,3,4)
    Cost?
    PDDL
    [] Exp Qwen2_72b: 
