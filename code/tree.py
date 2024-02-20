

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 读取数据集
file_path = r'D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv'
data = pd.read_csv(file_path)

# 选择一个比赛（match_id为示例值）
match_id = '2023-wimbledon-1701'
selected_match = data[data['match_id'] == match_id]

# 提取关键时间序列特征
time_series_features = ['set_no', 'game_no', 'point_no', 'server']

# 创建时间序列数据
time_series_data = selected_match[time_series_features]

# 按照时间顺序排序
time_series_data = time_series_data.sort_values(by=['set_no', 'game_no', 'point_no'])

# 特征工程
X = time_series_data[['set_no', 'game_no', 'point_no']]
y = time_series_data['winner']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林模型建立
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 模型评估
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')

# 时间序列可视化
plt.figure(figsize=(12, 6))

# 绘制比赛过程的折线图
plt.plot(time_series_data['point_no'], time_series_data['server'], label='Server')
plt.plot(time_series_data['point_no'], time_series_data['receiver'], label='Receiver')
plt.scatter(X_test['point_no'], y_pred, color='red', marker='x', label='Predicted Winner')

plt.title('Match Performance Time Series')
plt.xlabel('Point Number')
plt.ylabel('Player')
plt.legend()
plt.show()