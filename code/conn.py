import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
file_path = r'D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv'
Data = pd.read_csv(file_path)
data = Data[Data['match_id'] == '2023-wimbledon-1601']  # 选择第一场比赛
df = pd.DataFrame(data)
df['p1_score'] = df['p1_score'].replace('AD', '55')
df['p1_score'] = pd.to_numeric(df['p1_score'], errors='coerce')
df['p2_score'] = df['p2_score'].replace('AD', '55')
df['p2_score'] = pd.to_numeric(df['p2_score'], errors='coerce')
df['predict'] = 6- df['point_victor'].shift(-1).fillna(1.5) - df['point_victor'].shift(-2).fillna(1.5) - df['point_victor'].shift(-3).fillna(1.5) 
df['swings'] = 0
df['next_point_victor'] = 2-df['point_victor'].shift(-1).fillna(0)
now_swing = 0
for index in range(len(df['predict'])-1):
    if index == 0:
        if df['predict'].iloc[index] <=1:
            now_swing = -1
        elif df['predict'].iloc[index] >=2:
            now_swing = 1
    else:
        if df['predict'].iloc[index] == 3:
            now_swing = 1
        elif df['p1_games'].iloc[index] < df['p1_games'].iloc[index + 1]:
            now_swing = 1
        elif df['p1_sets'].iloc[index] < df['p1_sets'].iloc[index + 1]:
            now_swing = 1
        elif df['predict'].iloc[index] == 0:
            now_swing = -1
        elif df['p2_games'].iloc[index] < df['p2_games'].iloc[index + 1]:
            now_swing = -1
        elif df['p2_sets'].iloc[index] < df['p2_sets'].iloc[index + 1]:
            now_swing = -1
    df['swings'].iloc[index] = now_swing

df['swings'].iloc[index+1] = now_swing

# 添加新的列 'recent_points'，用于记录球员1近十球的得分数
df['score_difference'] = df['p1_score'].astype(str).str.replace('AD', '50').astype(int) - df['p2_score'].astype(str).str.replace('AD', '50').astype(int)
df['p1_total_distance_run'] = df['p1_distance_run'].cumsum()
df['p2_total_distance_run'] = df['p2_distance_run'].cumsum()
df['run_difference'] = df['p1_total_distance_run'] - df['p2_total_distance_run']
df['is_server'] = 2 - df['server']
data['p1_recent_points'] = (data['point_victor'] == 1).rolling(window=5,min_periods=0).sum()-(data['point_victor'] == 2).rolling(window=5,min_periods=0).sum()
data['p2_recent_points'] = (data['point_victor'] == 2).rolling(window=5,min_periods=0).sum()-(data['point_victor'] == 1).rolling(window=5,min_periods=0).sum()
df['p1_recent_points'] = data['p1_recent_points']
df['p2_recent_points'] = data['p2_recent_points']

df['p1_serve_count_last_5'] = (df['server'] == 1).astype(int).rolling(window=5, min_periods=1).sum()
df['score_difference_last_5'] = df['score_difference'].rolling(window=5, min_periods=1).sum()

df['p1_ace_count'] = df['p1_ace'].rolling(window=10, min_periods=1).sum()
df['p2_ace_count'] = df['p2_ace'].rolling(window=10, min_periods=1).sum()
df['p1_winner_count'] = df['p1_winner'].rolling(window=10, min_periods=1).sum()
df['p2_winner_count'] = df['p2_winner'].rolling(window=10, min_periods=1).sum()
df['p1_unf_err_count'] = df['p1_unf_err'].rolling(window=10, min_periods=1).sum()
df['p2_unf_err_count'] = df['p2_unf_err'].rolling(window=10, min_periods=1).sum()
df['p1_double_fault_count'] = df['p1_double_fault'].rolling(window=10, min_periods=1).sum()
df['p2_double_fault_count'] = df['p2_double_fault'].rolling(window=10, min_periods=1).sum()
df['p1_break_pt_count'] = df['p1_break_pt'].rolling(window=10, min_periods=1).sum()
df['p2_break_pt_count'] = df['p2_break_pt'].rolling(window=10, min_periods=1).sum()
df['p1_break_pt_won_count'] = df['p1_break_pt_won'].rolling(window=10, min_periods=1).sum()
df['p2_break_pt_won_count'] = df['p2_break_pt_won'].rolling(window=10, min_periods=1).sum()
df['three']=(10-df['point_victor'].shift(-1).fillna(0)-df['point_victor'].shift(-2).fillna(0)-df['point_victor'].shift(-3).fillna(0)-df['point_victor'].shift(-4).fillna(0)-df['point_victor'].shift(-5).fillna(0)-df['point_victor'].shift(-6).fillna(0)-df['point_victor'].shift(-7).fillna(0))>3
# 选择特征和目标

features = ['p1_serve_count_last_5','p2_score','p1_score','is_server','p1_recent_points','p2_recent_points','score_difference_last_5','run_difference','p1_ace_count','p2_ace_count','p1_winner_count','p2_winner_count','p1_unf_err_count','p2_unf_err_count','p1_double_fault_count','p2_double_fault_count','p1_break_pt_count','p2_break_pt_count','p1_break_pt_won_count','p2_break_pt_won_count']
target1 = 'swings'
data = data.fillna(0)
# 合并特征和目标
combinedata = df[features + [target1]+['next_point_victor']]

# data = data.dropna()
# 计算相关性矩阵（对于 'point_victor'）
correlation_matrix = combinedata.corr(method='spearman')

# 可视化相关性矩阵
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix for point_victor')
plt.tight_layout()
plt.show()
