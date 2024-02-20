import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from statsmodels.sandbox.stats.runs import runstest_1samp
# 生成示例数据
file_path = r'D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv'
Data = pd.read_csv(file_path)
# data = Data[Data['match_id'] == '2023-wimbledon-1701']  # 选择第一场比赛
data = Data[Data['player1'] == 'Carlos Alcaraz'] 
df = pd.DataFrame(data)
df['p1_score'] = df['p1_score'].replace('AD', '55')
df['p1_score'] = pd.to_numeric(df['p1_score'], errors='coerce')
df['p2_score'] = df['p2_score'].replace('AD', '55')
df['p2_score'] = pd.to_numeric(df['p2_score'], errors='coerce')

data['is_point_victor'] = 2 - data['point_victor']
# 添加新的列 'recent_points'，用于记录球员1近十球的得分数
data['p1_recent_points'] = (data['point_victor'] == 1).rolling(window=5,min_periods=0).sum()-(data['point_victor'] == 2).rolling(window=5,min_periods=0).sum()
data['p2_recent_points'] = (data['point_victor'] == 2).rolling(window=5,min_periods=0).sum()-(data['point_victor'] == 1).rolling(window=5,min_periods=0).sum()
df['p1_recent_points'] = data['p1_recent_points']
df['p2_recent_points'] = data['p2_recent_points']
# 对 'serve_width' 应用映射
df['serve_width'] = df['serve_width'].map({'W': 5, 'B/W': 5, 'B': 3, 'B/C': 2, 'C': 1})
# 对 'serve_depth' 应用映射
df['serve_depth'] = df['serve_depth'].map({'CTL': 2, 'NCTL': 1})
# 对 'return_depth' 应用映射
df['return_depth'] = df['return_depth'].map({'D': 2, 'ND': 1})
df['score_difference'] = df['p1_score'].astype(str).str.replace('AD', '50').astype(int) - df['p2_score'].astype(str).str.replace('AD', '50').astype(int)
df['p1_total_distance_run'] = df['p1_distance_run'].cumsum()
df['p2_total_distance_run'] = df['p2_distance_run'].cumsum()
df['run_difference'] = df['p1_total_distance_run'] - df['p2_total_distance_run']
df['is_server'] = 2 - df['server']



df['p1_serve_count_last_5'] = (df['server'] == 1).astype(int).rolling(window=5, min_periods=1).sum()
df['score_difference_last_5'] = df['score_difference'].rolling(window=5, min_periods=1).sum()



df['p1_ace_count'] = df['p1_ace'].rolling(window=10, min_periods=1).sum()
df['p2_ace_count'] = df['p2_ace'].rolling(window=10, min_periods=1).sum()
df['p1_winner_count'] = df['p1_winner'].rolling(window=10, min_periods=1).sum()
df['p2_winner_count'] = df['p2_winner'].rolling(window=10, min_periods=1).sum()
df['p1_unf_err_count'] = df['p1_unf_err'].rolling(window=10, min_periods=1).sum()
df['p2_unf_err_count'] = df['p2_unf_err'].rolling(window=10, min_periods=1).sum()
df['p1_break_pt_count'] = df['p1_break_pt'].rolling(window=10, min_periods=1).sum()
df['p2_break_pt_count'] = df['p2_break_pt'].rolling(window=10, min_periods=1).sum()
df['p1_break_pt_won_count'] = df['p1_break_pt_won'].rolling(window=10, min_periods=1).sum()
df['p2_break_pt_won_count'] = df['p2_break_pt_won'].rolling(window=10, min_periods=1).sum()

# 划分训练集和测试集
X = df[['p1_score','p2_score','is_server','p1_recent_points','p2_recent_points','score_difference_last_5','run_difference','p1_ace_count','p2_ace_count','p1_winner_count','p2_winner_count','p1_unf_err_count','p2_unf_err_count','p1_break_pt_count','p2_break_pt_count','p1_break_pt_won_count','p2_break_pt_won_count']]# 特征
df['predict'] = 6- df['point_victor'].shift(-1).fillna(1.5) - df['point_victor'].shift(-2).fillna(1.5) - df['point_victor'].shift(-3).fillna(1.5) 
df['next_point_victor'] = 2-df['point_victor'].shift(-1).fillna(0)
df['swings'] = 0
now_swing = 0
for index in range(len(df['predict'])-1):
    if index == 0:
        if df['predict'].iloc[index] <=1:
            now_swing = -1
        elif df['predict'].iloc[index] >=2:
            now_swing = 1
    else:
        if df['predict'].iloc[index] == 3 and df['point_victor'].iloc[index]==2:
            now_swing = 1
        elif df['p1_games'].iloc[index] < df['p1_games'].iloc[index + 1]:
            now_swing = 1
        elif df['p1_sets'].iloc[index] < df['p1_sets'].iloc[index + 1]:
            now_swing = 1
        elif df['predict'].iloc[index] == 0 and df['point_victor'].iloc[index]==1:
            now_swing = -1
        elif df['p2_games'].iloc[index] < df['p2_games'].iloc[index + 1]:
            now_swing = -1
        elif df['p2_sets'].iloc[index] < df['p2_sets'].iloc[index + 1]:
            now_swing = -1
    df['swings'].iloc[index] = now_swing

df['swings'].iloc[index+1] = now_swing

#使用 SimpleImputer 处理缺失值
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # 删除最后一行，因为 'next_point_victor' 的最后一行是 NaN
#标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_train, X_test, y_train, y_test = train_test_split(X_scaled[:],df['swings'], test_size=0.2, random_state=42)
print('111',X_scaled)
print(X)
def logistic_regression_predict(features, weights, intercept):
    # 计算线性组合
    linear_combination = np.dot(features, weights) + intercept
    
    # 计算概率
    probability = 1 / (1 + np.exp(-linear_combination))
    return probability>0.5


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
probabilities = logistic_regression_predict(X_scaled, weights, intercept)
probabilities2= 2*probabilities-1


plt.figure(figsize=(15, 8))

# 相关性分析
min_length = min(len(df['swings']), len(probabilities2))
swings = df['swings'].iloc[:min_length].values
probabilities2 = probabilities2[:min_length]
correlation, p_value = pearsonr(swings, probabilities2)
# Output the results
print("Correlation coefficient:", correlation)
print("P-value:", p_value)
print(f"Correlation: {correlation}")
swings_binary = np.sign(swings[:min_length])
probabilities2_binary = np.sign(probabilities2[:min_length])
jaccard_similarity = jaccard_score(swings_binary, probabilities2_binary)
print(f"Jaccard Similarity between Swings and Probabilities2: {jaccard_similarity}")
spearman_correlation,spearman_p_value= spearmanr(swings, probabilities2)
print(f"Spearman Correlation: {spearman_correlation},p_value: {spearman_p_value}")
kendall_tau,kendall_p_value = kendalltau(swings, probabilities2)
print(f"Kendall Tau: {kendall_tau},p_value: {kendall_p_value}")

# 游程检验
result_statistic,result_pvalue = runstest_1samp(df['swings'])

# 输出检验结果
print("Z值:", result_statistic )
print("P值:", result_pvalue)
# 绘制 swings
plt.plot(swings, marker='o', linestyle='-', color='b', label='St')

# # 绘制 predicted probabilities
plt.plot(probabilities2, marker='x', linestyle='-', color='r', label='Predicted St')

# 设置标题和标签
# plt.title('True Swings and Predicted Swings Over Time')

plt.xlabel('points')
plt.ylabel('Values')
plt.grid(True)
plt.legend()

# 显示图形
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(probabilities2, swings)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()




