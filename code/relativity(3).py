import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# 加载数据
file_path = r'D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv'
Data = pd.read_csv(file_path)
data = Data[Data['player1'] == 'Carlos Alcaraz']  # 选择第一场比赛
data = data[Data['match_id'] == '2023-wimbledon-1401']
df = pd.DataFrame(data)
df['swings']= 0
#真实转折点
current_swing = 0

for index in range(len(df)-2):
    if index == 0:
        if df['point_victor'].iloc[index] == 1:
            current_swing = 1
        elif df['point_victor'].iloc[index] == 2:
            current_swing = 0
    if df['p1_games'].iloc[index] < df['p1_games'].iloc[index + 1]:
        current_swing = 1
    elif df['p2_games'].iloc[index] < df['p2_games'].iloc[index + 1]:
        current_swing = 0
    elif df['p1_sets'].iloc[index] < df['p1_sets'].iloc[index + 1]:
        current_swing = 1
    elif df['p2_sets'].iloc[index] < df['p2_sets'].iloc[index + 1]:
        current_swing = 0
    elif df['p1_points_won'].iloc[index-1] == df['p1_points_won'].iloc[index] and \
         df['p1_points_won'].iloc[index] < df['p1_points_won'].iloc[index + 1] and \
         df['p1_points_won'].iloc[index+1] < df['p1_points_won'].iloc[index + 2] and \
         df['p1_points_won'].iloc[index+2] < df['p1_points_won'].iloc[index+3]:
        current_swing = 1
    elif df['p2_points_won'].iloc[index-1] == df['p2_points_won'].iloc[index] and \
         df['p2_points_won'].iloc[index] < df['p2_points_won'].iloc[index + 1] and \
         df['p2_points_won'].iloc[index+1] < df['p2_points_won'].iloc[index+2] and \
         df['p2_points_won'].iloc[index+2] < df['p2_points_won'].iloc[index+3]:
        current_swing = 0

    df.at[index, 'swings'] = current_swing

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



df['predict_winner_length']=(10-df['point_victor'].shift(-1).fillna(1.5)-df['point_victor'].shift(-2).fillna(1.5)-df['point_victor'].shift(-3).fillna(1.5))-df['point_victor'].shift(-4).fillna(1.5)-df['point_victor'].shift(-5).fillna(1.5)>2
print(df['predict_winner_length'])
# 选择特征和目标
features = ['p1_recent_points','p2_recent_points','is_server','total_score_difference','p1_serve_count','p2_serve_count','distance_difference','p1_ace_count','p2_ace_count','winner_count_difference','p1_winner_count','p2_winner_count','p1_unf_err_count','p2_unf_err_count','p1_break_pt_count','p2_break_pt_count','p1_break_pt_won_count','p2_break_pt_won_count']
target1 = 'predict_winner_length'
target2 = 'swings'
data = data.fillna(0)
# 合并特征和目标
data = df[features + [target1]+[target2]]

# data = data.dropna()
# 计算相关性矩阵（对于 'point_victor'）
correlation_matrix = data.corr(method='spearman')

# 可视化相关性矩阵
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix for point_victor')
plt.tight_layout()
plt.show()
