import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
file_path = r'D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv'
Data = pd.read_csv(file_path)
data = Data[Data['match_id'] == '2023-wimbledon-1701']  # 选择第一场比赛
print(data)
# 初始化变量
momentum_data = [0] * len(data)  # 存储动量数据
momentum_data2 = [0] * len(data)
t =[0] * len(data)
t2 = [0] * len(data)
con = 0
con2 = 0
points = [0] * len(data) 
points2 = [0] * len(data)

# 遍历每个比赛点
for i in range(6950,len(data)+6950):
    if data.loc[i, 'point_victor'] == 1:
        points[i-6950] = (1 + (2-data.loc[i, 'server'])*0.2)*(1+con*0.2)*(1+0.2*data.loc[i, 'p1_ace'])*(1+0.2*data.loc[i, 'p1_winner'])*(1+0.2*data.loc[i, 'p1_double_fault'])*(1+0.2*data.loc[i, 'p1_unf_err'])
        con+=1
    else:
        con=0
        points[i-6950] = -(1 + (2-data.loc[i, 'server'])*0.2)*(1+con*0.2)*(1+0.2*data.loc[i, 'p2_ace'])*(1+0.2*data.loc[i, 'p2_winner'])*(1+0.2*data.loc[i, 'p2_double_fault'])*(1+0.2*data.loc[i, 'p2_unf_err'])

    # 获取当前比赛点的信息
    t[i-6950] = data.loc[i, 'elapsed_time']
    if i==6950:
        momentum_data[i-6950] =0
    else:
        # if data.loc[i-1, 'point_victor'] == 1:
        #     momentum_data[i] =momentum_data[i-1]*0.8 +1+data.loc[i-1, 'p1_ace']+data.loc[i-1, 'p1_winner']
        # else:
        #     momentum_data[i] =momentum_data[i-1]*0.8 -1-data.loc[i-1, 'p2_ace']-data.loc[i-1, 'p2_winner']
        if i>=6961:
            # momentum_data[i-6950] = points[i-5-6950]*0.2+points[i-4-6950]*0.4+points[i-3-6950]*0.6+points[i-2-6950]*0.8+points[i-1-6950]
            momentum_data[i-6950] = momentum_data[i-1-6950]-points[i-11-6950]+points[i-1-6950]
        else:
            momentum_data[i-6950] = momentum_data[i-1-6950] + points[i-1-6950]
               
for i in range(6950,len(data)+6950):
    if data.loc[i, 'point_victor'] == 2:
        points2[i-6950] = (1 + (2-data.loc[i, 'server'])*0.2)*(1+con*0.2)*(1+0.2*data.loc[i, 'p2_ace'])*(1+0.2*data.loc[i, 'p2_winner'])*(1+0.2*data.loc[i, 'p2_double_fault'])*(1+0.2*data.loc[i, 'p2_unf_err'])
        con+=1
    else:
        con=0
        points2[i-6950] = -(1 + (2-data.loc[i, 'server'])*0.2)*(1+con*0.2)*(1+0.2*data.loc[i, 'p1_ace'])*(1+0.2*data.loc[i, 'p1_winner'])*(1+0.2*data.loc[i, 'p1_double_fault'])*(1+0.2*data.loc[i, 'p1_unf_err'])

    # 获取当前比赛点的信息
    t2[i-6950] = data.loc[i, 'elapsed_time']
    if i==6950:
        momentum_data2[i-6950] =0
    else:
        # if data.loc[i-1, 'point_victor'] == 1:
        #     momentum_data[i] =momentum_data[i-1]*0.8 +1+data.loc[i-1, 'p1_ace']+data.loc[i-1, 'p1_winner']
        # else:
        #     momentum_data[i] =momentum_data[i-1]*0.8 -1-data.loc[i-1, 'p2_ace']-data.loc[i-1, 'p2_winner']
        if i>=6961:
            # momentum_data[i-6950] = points[i-5-6950]*0.2+points[i-4-6950]*0.4+points[i-3-6950]*0.6+points[i-2-6950]*0.8+points[i-1-6950]
            momentum_data2[i-6950] = momentum_data2[i-1-6950]-points2[i-11-6950]+points2[i-1-6950]
        else:
            momentum_data2[i-6950] = momentum_data2[i-1-6950] + points2[i-1-6950]
               
x = range(len(momentum_data))
# 可视化动量变化
plt.figure(figsize=(12, 6))
plt.plot(t, momentum_data, label='Momentum', color='blue')
plt.plot(t2, momentum_data2, label='Momentum2', color='green')
plt.title('Momentum in Tennis Match')
plt.xlabel('Elapsed Time (minutes)')  # 见完整版
plt.ylabel('Momentum')
plt.legend()
# 添加数值标签
for i, txt in enumerate(momentum_data):
    plt.text(t[i], momentum_data[i], f'{txt:.2f}', fontsize=8, ha='left', va='bottom')

plt.show()
