import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# 生成示例数据
file_path = r'D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv'
Data = pd.read_csv(file_path)
data = Data[Data['match_id'] == '2023-wimbledon-1701']  # 选择第一场比赛
df = pd.DataFrame(data)

# 添加分差和当前总跑动距离
df['point_difference'] = df['p1_score'].astype(str).str.replace('AD', '50').astype(int) - df['p2_score'].astype(str).str.replace('AD', '50').astype(int)
df['total_distance_run'] = df['p1_distance_run'].cumsum()
df['is_server'] = 2-df['server']
# 划分训练集和测试集
X = df[['is_server', 'p1_ace', 'p1_winner', 'p1_break_pt', 'p1_break_pt_won', 'p1_unf_err', 'p1_double_fault', 'point_difference', 'total_distance_run','speed_mph']]# 特征

y = 2 - df['point_victor']  # 下一次的结果

# 使用 SimpleImputer 处理缺失值
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # 删除最后一行，因为 'next_point_victor' 的最后一行是 NaN
# # 标准化特征
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_imputed)
X_train, X_test, y_train, y_test = train_test_split(X_imputed,y, test_size=0.2, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# 获取学到的权重和截距项
weights = model.coef_[0]
intercept = model.intercept_[0]
print(f'Learned Weights: {weights}')
print(f'Intercept: {intercept}')

# 计算模型对每个样本的预测概率
probabilities = model.predict_proba(X_imputed)[:, 1]

# 计算模型对每个样本的预测标签
predictions = model.predict(X_imputed)

# 计算势头函数
momentum = intercept + np.dot(X_imputed, weights)

# 绘制势头函数和真实势头
plt.figure(figsize=(12, 6))
plt.plot(momentum, label='Predicted Momentum', color='blue')
plt.plot(3 * y[:-1] - 1.5, label='True Momentum', color='black')  # 删除最后一行对应的 y
plt.title('Momentum Function')
plt.xlabel('Data Points')
plt.ylabel('Momentum')
plt.legend()
plt.show()
