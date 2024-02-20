import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 生成示例数据
file_path = r'D:\VScodeWorkforce\LaTex\美赛\code\Wimbledon_featured_matches.csv'
Data = pd.read_csv(file_path)
data = Data[Data['match_id'] == '2023-wimbledon-1301'] 
df = pd.DataFrame(data)

data['is_point_victor'] = 2 - data['point_victor']

length = 10
# Adding new column 'recent_points' to record player 1's recent points
df['p1_recent_points'] = (data['point_victor'] == 1).rolling(window=length, min_periods=0).sum() - (data['point_victor'] == 2).rolling(window=length, min_periods=0).sum()
df['p2_recent_points'] = (data['point_victor'] == 2).rolling(window=length, min_periods=0).sum() - (data['point_victor'] == 1).rolling(window=length, min_periods=0).sum()

# Mapping serve_width and replacing values in serve_depth and return_depth
df['serve_width'] = df['serve_width'].map({'W': 5, 'B/W': 4, 'B': 3, 'B/C': 2, 'C': 1})
df['serve_depth'].replace({'CTL': 2, 'NCTL': 1}, inplace=True)
df['return_depth'].replace({'D': 2, 'ND': 1}, inplace=True)

# Calculating score_difference, cumulative distance run, and is_server
df['score_difference'] = df['p1_score'].astype(str).str.replace('AD', '50').astype(int) - df['p2_score'].astype(str).str.replace('AD', '50').astype(int)
df['p1_total_distance_run'] = df['p1_distance_run'].cumsum()
df['p2_total_distance_run'] = df['p2_distance_run'].cumsum()
df['is_server'] = 2 - df['server']

# Rolling calculations with the window size set to length
df['p1_total_distance_run'] = df['p1_total_distance_run'].rolling(window=length, min_periods=1).sum()
df['p2_total_distance_run'] = df['p2_total_distance_run'].rolling(window=length, min_periods=1).sum()
df['distance_difference'] = df['p1_total_distance_run'] - df['p2_total_distance_run']
df['p1_serve_count'] = (df['server'] == 1).astype(int).rolling(window=length, min_periods=1).sum()
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



df['predict_winner_length3']=(10-df['point_victor'].shift(-1).fillna(1.5)-df['point_victor'].shift(-2).fillna(1.5)-df['point_victor'].shift(-3).fillna(1.5)-df['point_victor'].shift(-4).fillna(1.5)-df['point_victor'].shift(-5).fillna(1.5))>2

print(df['predict_winner_length3'])

# 划分训练集和测试集
X = df[['is_server','total_score_difference','p1_serve_count','distance_difference','p1_ace_count','p2_ace_count','winner_count_difference','p1_winner_count','p2_winner_count','p1_unf_err_count','p2_unf_err_count','p1_break_pt_count','p2_break_pt_count','p1_break_pt_won_count','p2_break_pt_won_count']]# 特征

y = df['predict_winner_length3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple neural network model
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)