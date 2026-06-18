# 模型评估分析笔记(供最终写 README 用)

> 这份笔记记录 Project-2 训练阶段的关键实验数据、对比表格和分析思考。
> 最终写 README 时,从这里取用,整理成「Model Evaluation & Analysis」章节。

---

## 一、背景:类别不平衡

- 数据集 `liked` 正样本(点赞)仅占约 **12%**,负样本(不点赞)约 88%。
- 这是严重的**类别不平衡(class imbalance)**。
- 后果:如果用 accuracy 评估,「全部预测不点赞」就能拿到 ~88% 的 accuracy,
  但这种模型毫无用处(一个会点赞的用户都识别不出)。

---

## 二、两轮实验对比

### 第 1 轮:固定参数,无 class_weight,只看 accuracy(躺平基线)

四个模型用固定参数训练(无 class_weight、无调参),只用 accuracy 评估。
结果四个模型 accuracy **完全相同**,全部收敛到 **0.8803**:

| 模型 | Accuracy |
|---|---|
| Logistic_Regression | 0.8803 |
| Random_Forest | 0.8803 |
| Gradient_Boosting | 0.8803 |
| LightGBM | 0.8803 |

(全量 500000 行结果。最佳模型按 accuracy 选,因四者相同,
实际由浮点高位小数差异决定,无实质意义——当时被选为 Gradient_Boosting。)

**为什么四个完全相同?**
所有模型都「躺平」预测多数类(几乎全猜 0 / 不点赞),
预测结果几乎一致,所以 accuracy 都 ≈ 多数类基线(88%)。
- accuracy ≈ 88% 的负样本全部猜对 / 100% ≈ 0.88
- 此时 Recall ≈ 0(一个点赞用户都没抓到)
- **结论:在不平衡数据上,accuracy 完全无区分度,
  四个本质不同的模型得到相同分数,根本选不出有意义的最佳模型。**

这组「躺平基线」是后续所有改进的起点和对照组:
它证明了「只用 accuracy + 不处理不平衡」这条路走不通,
从而引出第 2 轮的 class_weight + 多指标 + F1 方案。

### 第 2 轮:GridSearch 调参 + class_weight='balanced' + 多指标

(注:此表为 20000 行采样的验证结果,用于展示方法论;
最终 README 应替换为全量 500000 行的结果。)

| 模型 | F1 | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| **Logistic_Regression** | **0.2022** | 0.5089 | 0.5128 | 0.1269 | 0.4980 |
| Random_Forest | 0.1888 | 0.5275 | 0.7250 | 0.1488 | 0.2581 |
| Gradient_Boosting | 0.0118 | 0.5304 | 0.8748 | 0.2727 | 0.0060 |
| LightGBM | 0.1775 | 0.5150 | 0.7035 | 0.1353 | 0.2581 |

各模型 GridSearch 选出的最优参数:
- Logistic_Regression: `{'C': 10.0}`
- Random_Forest: `{'max_depth': 8, 'n_estimators': 100}`
- Gradient_Boosting: `{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}`
- LightGBM: `{'learning_rate': 0.05, 'n_estimators': 100, 'num_leaves': 31}`

---

## 三、核心发现(README 重点)

### 发现 1:指标选择直接改变「最佳模型」

同一批模型,换一个评估指标,冠军就换人:

| 选择依据 | 选出的冠军 | 该模型的 Recall | 说明 |
|---|---|---|---|
| **Accuracy** | Gradient_Boosting (0.8748) | **0.006** | 看着分高,实则几乎不抓点赞用户(躺平) |
| **F1** | Logistic_Regression (0.2022) | **0.498** | 真的抓到约一半点赞用户 |

→ 结论:**在不平衡数据上,用 accuracy 选模型会选出「伪装成高分的躺平模型」。**
F1 同时约束 Precision 和 Recall,能识破并淘汰躺平模型。

### 发现 2:class_weight 让模型「活过来」

- 无 class_weight:所有模型 Recall ≈ 0(躺平猜多数类)
- 加 class_weight='balanced':模型被迫重视少数类,Recall 从 0 涨到 0.25~0.50
- 代价:accuracy 下降(模型牺牲多数类正确率,换取抓住少数类)
- 例外:sklearn 的 GradientBoostingClassifier 不支持 class_weight,
  所以它仍躺平(Recall=0.006),正好成为对比组,反衬 F1 的价值。

### 发现 3:特征信号本身很弱(诚实的局限)

- 所有模型 AUC 都在 0.51~0.53,接近 0.5(随机水平)。
- 说明给定的 10 个特征(时间、活跃度、热度、SVD embedding)
  对「预测点赞」的信号本身就很弱,无论怎么调参/平衡都难以突破。
- 这是数据本身的局限,符合本项目「工程链路练习」的定位
  (重点是跑通端到端系统,不是刷模型性能)。

---

## 四、为什么选 F1 作为主依据(决策理由)

- Accuracy:不平衡下失真,排除。
- Recall 单用:会选出「全猜点赞」的极端模型。
- Precision 单用:会选出「过度保守、几乎不预测点赞」的模型。
- **F1**:Precision + Recall 的调和平均,同时约束两者,堵死两种极端躺平;
  贴合 project.md 的二分类判定框架;面试易解释。→ 选为主依据。
- AUC:衡量排序能力,对不平衡稳健,与推荐系统「排序」本质契合。
  本项目先记录、暂不作主依据,留待后续研究后可切换。

(选择依据经导师确认:可使用 F1/AUC 等更适合不平衡数据的指标,不必拘泥 accuracy。)

---

## 五、关于「项目目的」的思考(可写进 README 的洞察)

两种推荐系统哲学:
- **返回准确概率**:关心校准(calibration),模型说 30% 就真有 30% 会点赞。
  → project.md 前端展示「概率 + 置信度」属于这一类。
- **最大化抓住会喜欢的去推送**:关心 Recall / 排序(AUC)。
  → 真实短视频推荐(抖音/YouTube)的核心诉求。

按 project.md 字面目的(跑通系统、返回概率),指标选择不敏感;
按真实推荐目的(抓住喜欢的去推),应重点看 Recall/AUC。
本项目选 F1 是居中稳妥的工程选择。

---

## 六、后续可做的进阶(bonus,README 可列为 Future Work)

- 对 GradientBoosting 用 sample_weight 变通实现类似 class_weight 的效果。
- 引入混淆矩阵可视化。
- 深入研究 AUC,考虑改用 AUC 或 Recall 作为主选择依据。
- 尝试更强的特征工程,突破当前 AUC≈0.5 的瓶颈。
