import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 加载数据
file_path = r"D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv"
Data = pd.read_csv(file_path)
data = Data[Data['match_id'] == '2023-wimbledon-1701']  # 选择第一场比赛

data['is_point_victor'] = 2 - data['point_victor']
# 添加新的列 'recent_points'，用于记录球员1近十球的得分数
data['p1_recent_points'] = (data['point_victor'] == 1).rolling(window=10,min_periods=0).sum()-(data['point_victor'] == 2).rolling(window=10,min_periods=0).sum()
data['p2_recent_points'] = (data['point_victor'] == 2).rolling(window=10,min_periods=0).sum()-(data['point_victor'] == 1).rolling(window=10,min_periods=0).sum()
data['total_points_difference'] = data['p1_points_won'] - data['p2_points_won']
data['total_ace_difference'] = data['p1_ace'].cumsum() - data['p2_ace'].cumsum()
data['total_winner_difference'] = data['p1_winner'].cumsum() - data['p2_winner'].cumsum()
data['total_break_pt_difference'] = data['p1_break_pt'].cumsum() - data['p2_break_pt'].cumsum()
data['total_break_pt_won_difference'] = data['p1_break_pt_won'].cumsum() - data['p2_break_pt_won'].cumsum()
data['total_unf_err_difference'] = data['p1_unf_err'].cumsum() - data['p2_unf_err'].cumsum()
data['total_double_fault_difference'] = data['p1_double_fault'].cumsum() - data['p2_double_fault'].cumsum()

data['p1_total_serve_count_recent'] = (data['server']==1).rolling(window=10,min_periods=0).sum()
data['p2_total_serve_count_recent'] = (data['server']==2).rolling(window=10,min_periods=0).sum()
data['total_serve_count_difference_recent'] = data['p1_total_serve_count_recent'] - data['p2_total_serve_count_recent']
data['p1_total_ace_recent']=data['p1_ace'].rolling(window=10,min_periods=0).sum()
data['p2_total_ace_recent']=data['p2_ace'].rolling(window=10,min_periods=0).sum()
data['p1_total_winner_recent']=data['p1_winner'].rolling(window=10,min_periods=0).sum()
data['p2_total_winner_recent']=data['p2_winner'].rolling(window=10,min_periods=0).sum()
data['p1_total_break_pt_recent']=data['p1_break_pt'].rolling(window=10,min_periods=0).sum()
data['p2_total_break_pt_recent']=data['p2_break_pt'].rolling(window=10,min_periods=0).sum()
data['p1_total_break_pt_won_recent']=data['p1_break_pt_won'].rolling(window=10,min_periods=0).sum()
data['p2_total_break_pt_won_recent']=data['p2_break_pt_won'].rolling(window=10,min_periods=0).sum()
data['p1_total_unf_err_recent']=data['p1_unf_err'].rolling(window=10,min_periods=0).sum()
data['p2_total_unf_err_recent']=data['p2_unf_err'].rolling(window=10,min_periods=0).sum()
data['p1_total_double_fault_recent']=data['p1_double_fault'].rolling(window=10,min_periods=0).sum()
data['p2_total_double_fault_recent']=data['p2_double_fault'].rolling(window=10,min_periods=0).sum()
data['p1_total_distance_run_recent']=data['p1_distance_run'].rolling(window=10,min_periods=0).sum()
data['p2_total_distance_run_recent']=data['p2_distance_run'].rolling(window=10,min_periods=0).sum()

print(data['p1_total_serve_count_recent'])
print(data['p2_total_serve_count_recent'])
data['score_difference'] = data['p1_score'].astype(str).str.replace('AD', '50').astype(int) - data['p2_score'].astype(str).str.replace('AD', '50').astype(int)
data['total_distance_run'] = data['p1_distance_run'].cumsum()
data['is_server'] = 2 - data['server']
data['is_point_victor'] = 2 - data['point_victor']

data['game_winner'] = 0
data['swings'] = 0
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
# 遍历每个小局，确定小局的最后一个球的获胜者，并将整个小局内的每个球的game_winner都设置为这个人
for index in range(len(data)):
    if index < len(data) - 1:
        if data['p1_games'].iloc[index] < data['p1_games'].iloc[index + 1] or (data['p1_games'].iloc[index] > data['p1_games'].iloc[index + 1] and data['p1_games'].iloc[index] == data['p2_games'].iloc[index] + 2):
            for index2 in range(index + 1):
                if data['game_winner'].iloc[index2] == 0:
                    data.at[index2, 'game_winner'] = 1
        elif data['p2_games'].iloc[index] < data['p2_games'].iloc[index + 1] or (data['p2_games'].iloc[index] > data['p2_games'].iloc[index + 1] and data['p2_games'].iloc[index] == data['p1_games'].iloc[index] + 2):
            for index2 in range(index + 1):
                if data['game_winner'].iloc[index2] == 0:
                    data.at[index2, 'game_winner'] = 2
    else:
        if data['p1_games'].iloc[index] > data['p2_games'].iloc[index]:
            for index2 in range(index + 1):
                if data['game_winner'].iloc[index2] == 0:
                    data.at[index2, 'game_winner'] = 1
        else:
            for index2 in range(index + 1):
                if data['game_winner'].iloc[index2] == 0:
                    data.at[index2, 'game_winner'] = 2
data['is_game_winner'] = 2 - data['game_winner']


data = data.fillna(0)
# 选择特征和目标
features1 = ['is_server','is_point_victor', 'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner', 'p1_break_pt', 'p2_break_pt', 'p1_break_pt_won', 'p2_break_pt_won', 'p1_unf_err', 'p2_unf_err', 'p1_double_fault', 'p2_double_fault', 'p1_games', 'p2_games']
features2 = ['total_distance_run', 'score_difference', 'total_points_difference', 'total_ace_difference', 'total_winner_difference', 'total_break_pt_difference', 'total_break_pt_won_difference', 'total_unf_err_difference', 'total_double_fault_difference']
features3 = ['p1_total_serve_count_recent', 'p2_total_serve_count_recent','total_serve_count_difference_recent', 'p1_total_ace_recent', 'p2_total_ace_recent', 'p1_total_winner_recent', 'p2_total_winner_recent', 'p1_total_break_pt_recent', 'p2_total_break_pt_recent', 'p1_total_break_pt_won_recent', 'p2_total_break_pt_won_recent', 'p1_total_unf_err_recent', 'p2_total_unf_err_recent', 'p1_total_double_fault_recent', 'p2_total_double_fault_recent', 'p1_total_distance_run_recent', 'p2_total_distance_run_recent']
target1 = 'point_victor'
target2 = 'next_point_victor'
target3 = 'is_game_winner'  # 新的目标变量
target4 = 'p1_recent_points'
target5 = 'p2_recent_points'
target6 = 'swings'
df = pd.DataFrame(data)

# 创建数据的副本
selected_data = data[features1+[target4]+[target5]+[target6]]

# selected_data[target2] = data[target1].shift(-1)
# selected_data = selected_data.dropna()

# pd.set_option("display.max_rows", None)
# print(selected_data['is_server'])
# print(selected_data[target1],selected_data[target2])
# pd.reset_option("display.max_rows")
# df1[target1] = 2 - df[target1]

# # 移动 'next_point_victor' 列，使其表示下一次的得分
# df1[target2] = df1[target1].shift(-1)

# df1[target3] = df['is_game_winner']

# 计算相关性矩阵

correlation_matrix = selected_data.corr(method='kendall')

# 可视化相关性矩阵
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix for Selected Features and momentum')
plt.tight_layout()
plt.show()
