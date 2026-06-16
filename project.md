## 1. 项目预览

**项目名称**：Cognitive Shorts Recommendation System

**背景**：平台存在用户与视频的交互数据（观看、点赞、评论、分享等），构建一个的端到端 ML 系统：

- 离线：对原始 CSV 做数据处理，生成训练用特征表；
- 训练：用特征表训练多种模型并选择最佳；
- 在线：前端页面收集少量输入 → 后端推导完整特征 → 调用模型输出互动概率与置信度

**核心目标**
1. 生成 `processed_interactions.csv`（训练用特征样本表）
2. 使用该表训练 4 个模型：逻辑回归、随机森林、梯度提升树、LightGBM
3. 提供一个首页（Single Prediction）能输入必要字段，展示预测概率、置信度、耗时

---

## **2. 范围（Scope）**

### 2.1 范围包含
- 数据读取：`users.csv`、`videos.csv`、`interactions.csv`
- 数据拼接：按 `user_id`、`video_id` 合并
- 特征工程：时间特征、用户活跃度、视频热度、SVD Embedding
- 输出特征表：`processed_interactions.csv`
- 模型训练与保存：读取 `processed_interactions.csv` 训练四类模型，保存最佳模型
- 在线推理：前端收集输入 → 后端构造模型特征 → 输出预测结果

### 2.2 技术栈
- 语言：Python
- 前端：Streamlit
- 后端：FastAPI
- 数据处理：Pandas / Numpy
- 模型：scikit-learn + LightGBM
- 数据校验：Pandera

---

## **3. 数据集说明**

### 3.1 输入数据
1. `users.csv`：用户画像/账号信息/统计
2. `videos.csv`：视频元数据/内容特征/统计
3. `interactions.csv`：交互流水（每行一次用户-视频行为）

### 3.2 输出数据
4. `processed_interactions.csv`：训练用特征表（每行一个样本）

---

## 4. 数据处理需求

### 4.1 总体流程
1. 读取 3 个原始 CSV
2. 数据拼接（以 interactions 为主表）
3. 时间特征提取
4. 计算用户活跃度 / 视频热度
5. 构建 user-video 稀疏矩阵并做 TruncatedSVD 得到用户 embedding
6. 拼回样本表、选择最终特征列
7. 输出 `processed_interactions.csv`

### 4.2 数据拼接（Join 规则）
- 主表：`interactions`
- 与用户表拼接：`on='user_id'`，`how='left'`
- 与视频表拼接：`on='video_id'`，`how='left'`
- 只取必要列以降低内存

### 4.3 时间特征提取（必须字段）
从 `interactions.timestamp` 提取：
- `hour`：0-23
- `day_of_week`：0-6（周一=0）
- `is_weekend`：是否周末（day_of_week ∈ {5,6} → 1）

### 4.4 用户活跃度（user_activity）定义
使用简化公式（可复现、易解释）：
- `user_activity = subscriber_count + total_watch_time_minutes`
- 缺失值策略：NaN → 0

### 4.5 视频热度（video_popularity）定义
使用简化公式：
- `video_popularity = view_count + like_count`
- 缺失值策略：NaN → 0

### 4.6 SVD Embedding 生成
**目标**：生成 5 维用户 embedding：`user_emb_0..user_emb_4`

**方法**：
1. 取样本表中的 `user_id`、`video_id`
2. 将 user_id、video_id 转为 category 编码（得到行列索引）
3. 生成稀疏矩阵 `M`：shape = [num_users, num_videos]
   - 如果存在交互，则 M[u, v] = 1（本项目采用 1 作为最简交互强度）
4. `TruncatedSVD(n_components=5, random_state=42)` 得到 `user_embeddings`
5. 把 `user_embeddings` 按 `user_id` merge 回样本表

### 4.7 输出表结构（processed_interactions.csv）
必须包含以下列（顺序可固定以便训练一致）：

- 特征 X（10列）  
  1) `hour`  
  2) `day_of_week`  
  3) `is_weekend`  
  4) `user_activity`  
  5) `video_popularity`  
  6) `user_emb_0`  
  7) `user_emb_1`  
  8) `user_emb_2`  
  9) `user_emb_3`  
  10) `user_emb_4`

- 标签 y（1列）  
  11) `liked`（从 interactions.liked 来，必须是 0/1）

### 4.8 数据处理验收标准（Data Acceptance Criteria）
- 能生成 `data/processed_interactions.csv`
- 生成文件非空，列名齐全且类型合理（hour 0-23，is_weekend 0/1，liked 0/1）
- 无明显缺失导致崩溃（允许少量 NaN，但最终输出需填 0 或合理值）

---

## 5. 模型训练需求

### 5.1 输入
- `data/processed_interactions.csv`

### 5.2 任务类型
- 二分类：预测 `liked`（0/1）

### 5.3 训练/验证划分
- `train_test_split(test_size=0.2, random_state=42)`（固定随机种子，保证可复现）

### 5.4 模型列表
1. Logistic Regression
2. Random Forest
3. Gradient Boosting Tree
4. LightGBM

### 5.5 评估指标
- Accuracy

### 5.6 输出
- 保存最佳模型到 `models/`（例如 `best_model.pkl`）
- 保存元信息（例如：模型名、准确率、特征列名、标签含义）

### 5.7 训练验收标准
- 脚本运行成功，输出每个模型的指标
- 能生成模型文件与元数据文件
- 可被后端加载并用于预测

---

## **6. 首页（Single Prediction）**

### 6.1 首页输入字段
页面只收集这 4 个输入：
1. `user_id`（字符串）
2. `video_id`（字符串）
3. `watch_time_seconds`（数字）
4. `hour_of_day`（0-23）

**为什么只收集4个**  
训练时有 10 个特征，但其中大部分：

- 可由后端根据当前时间推导（day_of_week、is_weekend）
- 可通过 user_id/video_id 从数据表或特征字典查到（user_activity、video_popularity、embedding）
因此无需用户手动输入 10 个字段，否则页面不可用也不符合真实产品形态

---

## **7. 后端推导特征需求**

### 7.1 推导逻辑总览
前端传入：`user_id, video_id, watch_time_seconds, hour_of_day`

后端要构造模型输入（10维）：
- `hour`：直接由 `hour_of_day` 映射
- `day_of_week`：由“当前服务器时间”推导
- `is_weekend`：由 day_of_week 推导
- `user_activity`：用 `user_id` 去 users 查
- `video_popularity`：用 `video_id` 去 videos 查
- `user_emb_0..4`：用 `user_id` 查离线 embedding

### 7.2 特征映射表
| 模型特征         | 来源             | 推导方式                     |
| ---------------- | ---------------- | ---------------------------- |
| hour             | 前端 hour_of_day | 直接赋值                     |
| day_of_week      | 服务器当前时间   | datetime.now().weekday()     |
| is_weekend       | day_of_week      | day_of_week >= 5 → 1 else 0  |
| user_activity    | users.csv        | dict[user_id]，不存在→0      |
| video_popularity | videos.csv       | dict[video_id]，不存在→0     |
| user_emb_0..4    | 离线 embedding   | dict[user_id]，不存在→0 向量 |

### 7.3 缺失策略
- user_id 不存在：`user_activity = 0`，embedding = 0 向量
- video_id 不存在：`video_popularity = 0`
- 仍需返回预测结果（不能报错导致页面不可用）

### 7.4 数据校验
- hour_of_day 必须 0-23
- watch_time_seconds 必须 ≥ 0
- 必须保证传入模型的列与训练一致（列名与顺序）

---

## **8. 首页展示需求**

### 8.1 调用流程
- 点击 Predict → 前端向后端 `/predict` 发送 JSON
- 后端返回 `probability + confidence + prediction`
- 前端展示：
  - 概率仪表盘（0~1）
  - Probability 数值
  - Confidence（Low/Medium/High）
  - Response Time（ms）

### 8.2 置信度规则
- probability ≥ 0.8 → High
- 0.5 ≤ probability < 0.8 → Medium
- < 0.5 → Low

---

## 9. 验收用例

**用例 1：正常预测**
- 输入：数据集中存在的 user_id/video_id
- 期望：返回 probability 0~1、页面正常展示

**用例 2：冷启动预测**
- 输入：user_id=unknown_user，video_id=unknown_video
- 期望：不报错，仍返回结果（大概率较低），耗时正常

**用例 3：非法输入**
- hour_of_day = 30 或 watch_time_seconds = -1
- 期望：后端校验失败并返回可读错误信息（前端展示 error）

