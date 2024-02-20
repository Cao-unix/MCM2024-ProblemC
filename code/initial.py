import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# 生成示例数据

file_path = r'D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv'
Data = pd.read_csv(file_path)
data = Data[Data['match_id'] == '2023-wimbledon-1301']  # 选择第一场比赛

target1 = 'point_victor'
target2 = 'next_point_victor'
data[target2] = data[target1].shift(-1)
data['is_server_next'] = 2-data['server'].shift(-1)
data['game_winner'] = 0  # 初始化新的目标变量
# 遍历每个小局，确定小局的最后一个球的获胜者，并将整个小局内的每个球的game_winner都设置为这个人
for index in range(len(data)):
    if index<len(data)-1:
        if data['p1_games'].iloc[index] < data['p1_games'].iloc[index + 1] or (data['p1_games'].iloc[index] > data['p1_games'].iloc[index+1] and data['p1_games'].iloc[index]==data['p2_games'].iloc[index]+2 ):
            for index2 in range(index + 1):
                if data['game_winner'].iloc[index2] == 0:
                    data.at[index2, 'game_winner'] = 1
        elif data['p2_games'].iloc[index] < data['p2_games'].iloc[index + 1] or  (data['p2_games'].iloc[index] > data['p2_games'].iloc[index+1] and data['p2_games'].iloc[index]==data['p1_games'].iloc[index]+2 ):
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


pd.set_option("display.max_rows", None)
print(data['game_winner'])
pd.reset_option("display.max_rows")


# 删除缺失值
data = data.dropna()
df = pd.DataFrame(data)
df['is_server'] = 2-df['server']
df['is_point_victor'] = 2-df['point_victor']
# 划分训练集和测试集
X = df[['is_server','is_point_victor','p1_ace', 'p2_ace', 'p1_winner', 'p2_winner', 'p1_break_pt', 'p2_break_pt', 'p1_break_pt_won', 'p2_break_pt_won', 'p1_unf_err', 'p2_unf_err', 'p1_double_fault', 'p2_double_fault', 'p1_games', 'p2_games']]  # 特征
y = 2-df['game_winner']  # 结果
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林模型建立
model = RandomForestClassifier(n_estimators=100, random_state=42)
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
momentum = intercept + weights[0]*X['is_server']+weights[1]*X['is_point_victor']+weights[2]*X['p1_ace']+weights[3]*X['p2_ace']+weights[4]*X['p1_winner']+weights[5]*X['p2_winner']+weights[6]*X['p1_break_pt']+weights[7]*X['p2_break_pt']+weights[8]*X['p1_break_pt_won']+weights[9]*X['p2_break_pt_won']+weights[10]*X['p1_unf_err']+weights[11]*X['p2_unf_err']+weights[12]*X['p1_double_fault']+weights[13]*X['p2_double_fault']+weights[14]*X['p1_games']+weights[15]*X['p2_games']
print(f'Intercept: {intercept}')
print(f'Momentum: {momentum}')
probabilities = model.predict_proba(X)[:, 1]


# # 添加数值标签
# for i, txt in enumerate(momentum):
#     plt.text( momentum[i], f'{txt:.2f}', fontsize=8, ha='left', va='bottom')

# plt.show()
# # 假设你已经有训练好的逻辑回归模型 model 和特征权重 weights
# # 使用你的模型预测概率
# probabilities = model.predict_proba(X_test)[:, 1]  # 获取正势头的概率，注意这里使用了测试集 X_test

# 定义阈值
threshold = 0.5

# 将概率转化为二元的正/负势头
momentum_function = np.where(probabilities >= threshold, 1, 0)

# 获取权重和截距项
weights = model.coef_[0]
intercept = model.intercept_[0]

# 计算决策边界
decision_boundary = -(intercept + weights[0] * X_test.iloc[:, 0]) / weights[1]

# 绘制势头函数和决策边界
plt.figure(figsize=(12, 6))
plt.plot(momentum, label='Predicted Probabilities', color='blue')
plt.plot(3*y-1.5, label='True Momentum', color='black')
# plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
# plt.step(range(len(momentum_function)), momentum_function, label='Momentum Function', color='green', where='post')
# plt.plot(X_test.iloc[:, 0], decision_boundary, label='Decision Boundary', color='orange')
plt.title('Momentum Function with Decision Boundary')
plt.xlabel('Data Points')
plt.ylabel('Momentum')
plt.legend()
plt.show()

# # 绘制混淆矩阵
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()