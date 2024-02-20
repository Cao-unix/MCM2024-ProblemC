import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# 生成示例数据
file_path = r'D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv'
Data = pd.read_csv(file_path)
# data = Data[Data['player1'] == 'Carlos Alcaraz']  # 选择第一场比赛
data = Data[Data['match_id'] == '2023-wimbledon-1401'] 
df = pd.DataFrame(data)

data['is_point_victor'] = 2 - data['point_victor']

length = 5
# Adding new column 'recent_points' to record player 1's recent points
df['p1_recent_points'] = (data['point_victor'] == 1).rolling(window=length, min_periods=0).sum() - (data['point_victor'] == 2).rolling(window=length, min_periods=0).sum()
df['p2_recent_points'] = (data['point_victor'] == 2).rolling(window=length, min_periods=0).sum() - (data['point_victor'] == 1).rolling(window=length, min_periods=0).sum()

# df['p1_consecutive_points'] = 0
# df['p2_consecutive_points'] = 0
# current_points = 0
# # Constructing the consecutive points factor for point_victor=1
# for index in range(len(df)):
#     if index > 0 and df['point_victor'].iloc[index] == 1 and df['point_victor'].iloc[index-1] == 1:
#         current_points += 1
#     else:
#         current_points = 0
#     df.at[index, 'p1_consecutive_points'] = current_points

# current_points = 0
# # Constructing the consecutive points factor for point_victor=2
# for index in range(len(df)):
#     if index > 0 and df['point_victor'].iloc[index] == 2 and df['point_victor'].iloc[index-1] == 2:
#         current_points += 1
#     else:
#         current_points = 0
#     df.at[index, 'p2_consecutive_points'] = current_points




# print(df['p1_consecutive_points'])
# print(df['p2_consecutive_points'])
df.dropna()
# Mapping serve_width and replacing values in serve_depth and return_depth
df['serve_width'] = df['serve_width'].map({'W': 5, 'B/W': 4, 'B': 3, 'B/C': 2, 'C': 1})
df['serve_depth'].replace({'CTL': 2, 'NCTL': 1}, inplace=True)
df['return_depth'].replace({'D': 2, 'ND': 1}, inplace=True)

# Handling 'AD' values and NaN values in scores
df['p1_score'] = df['p1_score'].replace({'AD': '50', np.nan: '0'}).astype(int)
df['p2_score'] = df['p2_score'].replace({'AD': '50', np.nan: '0'}).astype(int)
df['score_difference'] = df['p1_score'] - df['p2_score']

df['p1_total_distance_run'] = df['p1_distance_run'].cumsum()
df['p2_total_distance_run'] = df['p2_distance_run'].cumsum()
df['is_server'] = 2 - df['server']

# Rolling calculations with the window size set to length
df['p1_total_distance_run'] = df['p1_total_distance_run'].rolling(window=length, min_periods=1).sum()
df['p2_total_distance_run'] = df['p2_total_distance_run'].rolling(window=length, min_periods=1).sum()
df['distance_difference'] = df['p1_total_distance_run'] - df['p2_total_distance_run']
df['p1_serve_count'] = (df['server'] == 1).astype(int).rolling(window=length, min_periods=1).sum()
df['p2_serve_count'] = (df['server'] == 2).astype(int).rolling(window=length, min_periods=1).sum()
df['total_score_difference'] = df['score_difference'].rolling(window=length, min_periods=1).sum()

df['p1_ace_count'] = df['p1_ace'].rolling(window=length, min_periods=1).sum()
df['p2_ace_count'] = df['p2_ace'].rolling(window=length, min_periods=1).sum()
df['p1_winner_count'] = df['p1_winner'].rolling(window=length, min_periods=1).sum()
df['p2_winner_count'] = df['p2_winner'].rolling(window=length, min_periods=1).sum()
df['winner_count_difference'] = df['p1_winner_count'] - df['p2_winner_count']
df['p1_unf_err_count'] = df['p1_unf_err'].rolling(window=length, min_periods=1).sum()
df['p2_unf_err_count'] = df['p2_unf_err'].rolling(window=length, min_periods=1).sum()
df['p1_break_pt_count'] = df['p1_break_pt'].rolling(window=length, min_periods=1).sum()
df['p2_break_pt_count'] = df['p2_break_pt'].rolling(window=length, min_periods=1).sum()
df['p1_break_pt_won_count'] = df['p1_break_pt_won'].rolling(window=length, min_periods=1).sum()
df['p2_break_pt_won_count'] = df['p2_break_pt_won'].rolling(window=length, min_periods=1).sum()



df['predict_winner_length']=(10-df['point_victor'].shift(-1).fillna(1.5)-df['point_victor'].shift(-2).fillna(1.5)-df['point_victor'].shift(-3).fillna(1.5)-df['point_victor'].shift(-4).fillna(1.5)-df['point_victor'].shift(-5).fillna(1.5))>2
print(df['predict_winner_length'])

# 划分训练集和测试集
X = df[['p1_recent_points','p2_recent_points','is_server','total_score_difference','p1_serve_count','p2_serve_count','distance_difference','p1_ace_count','p2_ace_count','winner_count_difference','p1_winner_count','p2_winner_count','p1_unf_err_count','p2_unf_err_count','p1_break_pt_count','p2_break_pt_count','p1_break_pt_won_count','p2_break_pt_won_count']]# 特征

y = df['predict_winner_length']

y =  y # 将两列合并为一个新的分类标签  
next_y=-df['point_victor'].shift(-1).dropna()
# 使用 SimpleImputer 处理缺失值
#imputer = SimpleImputer(strategy='mean')
#X_imputed = imputer.fit_transform(X)  # 删除最后一行，因为 'next_point_victor' 的最后一行是 NaN
# 标准化特征
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_imputed)
X_train, X_test, y_train, y_test = train_test_split(X[:],y, test_size=0.1, random_state=42)

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

plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()