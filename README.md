
重点：需要让模型**主动思考“搜什么、搜够了吗、还缺啥”**，从而形成一个 *Agentic / Iterative RAG* 的闭环。

---

## 一、核心目标

> 让模型在有限轮数内（2–3 步）**自主规划检索与改写策略**，直到找到足以支持/反驳 claim 的证据集。

---

## 二、问题建模（MDP）

### 1️. 状态 Sₜ

系统在第 t 步的所有可观测信息：

```python
Sₜ = {
  claim_text,
  retrieved_evidence_t,   # 当前累计证据集合
  retrieval_metrics_t,    # 覆盖度、重复率、时间匹配度等（用来训练reward model打分）
  step_t,                 # 当前轮数 (t ≤ L)
  cost_so_far             # 已消耗的预算（efficiency可以最后考虑）
}
```

> 相当于“当前我知道了哪些事实、花了多少成本、还剩几步能搜什么”。

### 2️. 动作 Aₜ(WIP)

在每步中，LLM可执行的操作：

* search
* 生成/改写查询（query rewrite）
* 触发下一跳 sub-query （多实体分解）
* 终止 (stop action)：认为已足够

---

## 三、奖励设计

让奖励既能驱动方向，又能在短序列中传递信号。

### 1. 密集奖励 rₜ （Dense Reward）

帮助模型知道“这一轮搜得有没有进步”，避免只靠终局信号：

```
rₜ = α₁ * ΔCoverage + α₂ * ΔKeyHit - α₃ * ΔCost
```

* **ΔCoverage** ：检索到的新证据覆盖率提升
* **ΔKeyHit** ：命中主实体/时间/数字等关键约束的比例
* **ΔCost** ：本轮调用检索/LLM token 开销

> 例子：若本轮多找到了两条包含主实体的新闻段落，则 rₜ ≈ +0.2 ~ +0.3。

### 2. 终止奖励 R_final

在 Episode 结束时给予主要信号：

```
R_final = β₁ * Judge_Correct
         + β₂ * Recall@20
         - β₃ * Normalized_Cost
         - β₄ * StalenessPenalty
```

* **Judge_Correct** ：最终 LLM judge 输出与 gold label 一致（+1 / 0）
* **Recall@20** ：是否在 top-20 证据中命中 gold 句
* **Normalized_Cost** ：越省成本越好
* **StalenessPenalty** ：证据过时则扣分

---

## 四、算法选择与训练技巧（WIP）

### 1️. 强化学习算法选择

* **PPO（Proximal Policy Optimization）**：稳定
* **GRPO / Actor–Critic**：在长序列或多动作空间中更节省显存
* GSPO，DAPO...待调研

### 2️. SFT / 规则模板预热

* 先用已有高质量策略（人工模板或成功日志）做SFT：
  训练模型学会基本 query 生成模式。
* 进入 RL 阶段时从这些行为起步，防止起步阶段模型随便搜。

### 3️. KL 正则（保持分布稳定）

PPO 优化目标中加入 KL 约束：

```
L = L_PPO - λ * KL(π_θ || π_ref)
```

π_ref 是 SFT 模型分布；防止模型输出完全偏离可理解策略。

### 4️. 停止策略

* 设置 **最大轮数 L ≤ 3**。
* 若 Coverage 或 KeyHit 超过阈值（如 ≥ 0.8）提前终止。

---

## 五、训练数据来源(WIP)
* 数据schema设计


---

## 六、评估指标

| 维度    | 指标                        |
| ----- | ------------------------- |
| 召回质量  | 句级 Recall@K / Precision@K |
| 策略效率  | 平均检索轮数、成本 tokens          |
| 判定准确率 | LLM judge 最终 F1           |
| 稳定性   | 回报方差、KL 漂移量               |
