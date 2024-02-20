import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.sandbox.stats.runs import runstest_1samp

# 加载数据
file_path = r"D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv"
Data = pd.read_csv(file_path)
data = Data[Data['match_id'] == '2023-wimbledon-1301']  # 选择第一场比赛
data['swings'] = 0
data['p1_recent_points'] = (data['point_victor'] == 1).rolling(window=10,min_periods=0).sum()-(data['point_victor'] == 2).rolling(window=10,min_periods=0).sum()
current_swing = 0

for index in range(len(data)-2):
    
    if data['p1_games'].iloc[index] < data['p1_games'].iloc[index + 1]:
        current_swing = 1
    elif data['p2_games'].iloc[index] < data['p2_games'].iloc[index + 1]:
        current_swing = -1
    elif data['p1_sets'].iloc[index] < data['p1_sets'].iloc[index + 1]:
        current_swing = 1
    elif data['p2_sets'].iloc[index] < data['p2_sets'].iloc[index + 1]:
        current_swing = -1
    elif data['p1_points_won'].iloc[index-1] == data['p1_points_won'].iloc[index] and \
         data['p1_points_won'].iloc[index] < data['p1_points_won'].iloc[index + 1] and \
         data['p1_points_won'].iloc[index+1] < data['p1_points_won'].iloc[index + 2] and \
         data['p1_points_won'].iloc[index+2] < data['p1_points_won'].iloc[index+3]:
        current_swing = 1
    elif data['p2_points_won'].iloc[index-1] == data['p2_points_won'].iloc[index] and \
         data['p2_points_won'].iloc[index] < data['p2_points_won'].iloc[index + 1] and \
         data['p2_points_won'].iloc[index+1] < data['p2_points_won'].iloc[index+2] and \
         data['p2_points_won'].iloc[index+2] < data['p2_points_won'].iloc[index+3]:
        current_swing = -1

    data.at[index, 'swings'] = current_swing

pd.set_option("display.max_rows", None)
print(data['swings'])
pd.reset_option("display.max_rows")
# 创建新的变量 standard_victor
data['standard_victor'] = np.where(data['point_victor'] == 1, 1, -1)

# 绘制图形（仅包含部分数据）
start_index = 0  # 设置开始索引
end_index = 200    # 设置结束索引

plt.figure(figsize=(10, 6))
plt.plot(data['swings'], label='swings', color='tab:blue')
plt.plot(data['standard_victor'], label='standard_victor', color='tab:green')
plt.plot(data['p1_recent_points'], label='p1_recent_points', color='tab:red')
plt.title('Swings and Standard Victor')
plt.xlabel('Index')
plt.legend(loc='upper left')
plt.show()

#相关性分析
feature = ['swings', 'standard_victor']
target = 'p1_recent_points'
selected_data = data[feature+[target]]

correlation_matrix = selected_data.corr(method='kendall')

# 可视化相关性矩阵
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix for Selected Features and momentum')
plt.tight_layout()
plt.show()

# 进行游程检验
result_statistic, result_pvalue = runstest_1samp(data['swings'].iloc[start_index:end_index])
print(f"Runs test statistic: {result_statistic}")
print(f"Runs test p-value: {result_pvalue}")

