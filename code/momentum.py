import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
file_path = r'D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv'
Data = pd.read_csv(file_path)

# 选择第一场比赛
data = Data[Data['match_id'] == '2023-wimbledon-1301']

data['is_point_victor'] = 2 - data['point_victor']
# 添加新的列 'recent_points'，用于记录球员1近十球的得分数
data['p1_recent_points'] = (data['point_victor'] == 1).rolling(window=10,min_periods=0).sum()-(data['point_victor'] == 2).rolling(window=10,min_periods=0).sum()
data['p2_recent_points'] = (data['point_victor'] == 2).rolling(window=10,min_periods=0).sum()-(data['point_victor'] == 1).rolling(window=10,min_periods=0).sum()
# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(data['p1_recent_points'], label='Player 1 Recent Points')
plt.plot(data['p2_recent_points'], label='Player 2 Recent Points')
plt.title('Rolling Sum of Recent Points')
plt.xlabel('Data Index')
plt.ylabel('Recent Points (Rolling Sum)')
plt.legend()
plt.show()

