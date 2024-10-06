# TODO
- [x] ranged, heavy, light 不同是什么
- [x] task to actions
- [x] task assignment
- [x] some bugs
  - [x] ranged auto attack
- [x] 重构 `obs_dict` 格式，`obs_dict["blue"]["base"] = list(base)` -> `obs["blue"]["id"] = unit_dict`
- [ ] ~~重构任务分配，优先按 id 分配（记住上一时间步的分配）~~(不如按规则分配更优，即谁离得近给谁做)
- [x] 重写任务 `[Attack Enemy](unit_type, enemy_type)`, `[Build Building](building_type, building_loc)`, `[Harvest Mineral](mineral_loc)`
- [x] 优化任务更新（目标矿采完后或基地没了删除采矿任务）
- [ ] 写提示词
  - [x] 任务描述
  - [x] 游戏规则
  - [x] 任务空间和参数要求
  - [ ] 示例
    - [ ] 逆风局示例
- [ ] Exp.
  - [ ] low-level: CoT/ReAct/LLM
  - [ ] high-level: CoT/ReAct/LLM
    - [ ] Expert Rules
  - [ ] Ours: CoT/ReAct/LLM


## LLM PLAY RTS

- LLM output task format, `[task](params)`
    - `[Deploy Unit](unit_type, tgt_loc)`   # 持续性任务
    - `[Harvest Mineral](mineral_loc)`  # 持续任务，目标矿采完后再删除任务
    - `[Build Building](building_type, building_loc)`  # 一次性任务
    - `[Produce Unit](produce_type, direction)`  # 一次性任务
    - `[Attack Enemy](unit_type, enemy_type)`  # 一次性任务

## Baselines

### Rule

- [2L](https://rubensolv.github.io/LocalLearnerIJCAI/)：引入了局部学习器 （2L），一种基于 PSRO 框架的学习算法。它主动选择一组参考策略来改进搜索信号，并定义元策略，这些元策略在元策略支持的策略数量方面“介于”IBR 和 FP 定义的元策略之间。
- POLighthRush：训练一名工人并让其收集资源。一旦有足够的资源建造兵营，就建造一个兵营。从那一刻起，不断训练轻型单位，并立即派遣它们攻击最近的敌方单位
- POWorkerRush：不断训练工人，让其中一个工人收集资源，并派遣所有其他工人立即攻击最近的敌方单位
- NaiveMCTS：将 NaiveMonteCarlo 的朴素采样思想与 MCTS 相结合
- RandomBiasedAI：随机执行移动，但强烈偏向于攻击、收获和返回（概率增加 5 倍）

### LLM ACT with low-level action

LLM/CoT/ReAct

# 问题

- [ ] 正在建造的 object 挡路
- [x] units 都没有 ID，不知道谁是谁
- [x] ！！！玩家资源 obs 一直是 0（环境问题）
- [x] 环境在游戏结束时自动 reset，导致无法的到最后一步的 obs，导致 metric 记录不准确

# 1009
## SijiaCui
- [ ] Abstarct refine
- [ ] Method 3.3\3.4
- [ ] 4.Skill-RTS

## ShuaiXu
- [ ] reflection: selection of expert tips examples
- [ ] Qwen2

## AiyaoHe
- [ ] Related Work, parameterized skill

## All
- [ ] OpenReview.net

# 1006
## SijiaCui
- [ ] Paper: Formulation/Method/Framework
- [ ] Check Env

## ShuaiXu
- [ ] two player? LLMs vs LLMs -> (reasoning cability)
- [ ] different LLMs vs Random/RandomBiased/... (videos record/logs), 3/5 matches, metrics?

## AiyaoHe
- [ ] Paper: Related Work

# 1003
## SijiaCui
- [ ] Paper: Abstract
- [ ] Paper: Method
- [ ] highlight the point in the figure -- skill lib

## ShuaiXu
- [ ] Exp prompt: {manual}, {examples}, [opponent]
- [ ] Exp Baseline: low-level: CoT/ReAct(examples)

## AiyaoHe
- [ ] Paper: Related Work

## Others
    High-level: TextSC2 adaptation, -> XXX --expertRule--> Parameter, 
    Harvest Mineral -> expertRule --> 1,2,3,4
    Oracel
    Harvest Mineral(1,2,3,4)
    Cost?
    PDDL
    [] Exp Qwen2_72b: 
