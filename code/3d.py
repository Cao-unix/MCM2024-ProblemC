import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# Load the example data
file_path = r'file_path'
Data = pd.read_csv(file_path)
data = Data[Data['match_id'] == '2023-wimbledon-1701']

# Replace 'AD' with '55' and convert scores to numeric
data['p1_score'] = data['p1_score'].replace('AD', '55')
data['p1_score'] = pd.to_numeric(data['p1_score'], errors='coerce')
data['p2_score'] = data['p2_score'].replace('AD', '55')
data['p2_score'] = pd.to_numeric(data['p2_score'], errors='coerce')

# Create a column 'is_point_victor' indicating the player who won the point
data['is_point_victor'] = 2 - data['point_victor']

# Add a new column 'recent_points' to record the recent points scored by player 1 and player 2
data['p1_recent_points'] = (data['point_victor'] == 1).rolling(window=5, min_periods=0).sum() - (data['point_victor'] == 2).rolling(window=5, min_periods=0).sum()
data['p2_recent_points'] = (data['point_victor'] == 2).rolling(window=5, min_periods=0).sum() - (data['point_victor'] == 1).rolling(window=5, min_periods=0).sum()

# Quantify non-numeric data
data['serve_width'] = data['serve_width'].map({'W': 5, 'B/W': 5, 'B': 3, 'B/C': 2, 'C': 1})
data['serve_depth'] = data['serve_depth'].map({'CTL': 2, 'NCTL': 1})
data['return_depth'] = data['return_depth'].map({'D': 2, 'ND': 1})

# Calculate the score difference
data['score_difference'] = data['p1_score'].astype(str).str.replace('AD', '50').astype(int) - data['p2_score'].astype(str).str.replace('AD', '50').astype(int)

# Calculate cumulative distance run and run difference
data['p1_total_distance_run'] = data['p1_distance_run'].cumsum()
data['p2_total_distance_run'] = data['p2_distance_run'].cumsum()
data['run_difference'] = data['p1_total_distance_run'] - data['p2_total_distance_run']

# Create a column 'is_server' indicating the serving player
data['is_server'] = 2 - data['server']

# Calculate various rolling statistics
data['p1_serve_count_last_5'] = (data['server'] == 1).astype(int).rolling(window=5, min_periods=1).sum()
data['score_difference_last_5'] = data['score_difference'].rolling(window=5, min_periods=1).sum()
data['p1_ace_count'] = data['p1_ace'].rolling(window=10, min_periods=1).sum()
data['p2_ace_count'] = data['p2_ace'].rolling(window=10, min_periods=1).sum()
data['p1_winner_count'] = data['p1_winner'].rolling(window=10, min_periods=1).sum()
data['p2_winner_count'] = data['p2_winner'].rolling(window=10, min_periods=1).sum()
data['p1_unf_err_count'] = data['p1_unf_err'].rolling(window=10, min_periods=1).sum()
data['p2_unf_err_count'] = data['p2_unf_err'].rolling(window=10, min_periods=1).sum()
data['p1_break_pt_count'] = data['p1_break_pt'].rolling(window=10, min_periods=1).sum()
data['p2_break_pt_count'] = data['p2_break_pt'].rolling(window=10, min_periods=1).sum()
data['p1_break_pt_won_count'] = data['p1_break_pt_won'].rolling(window=10, min_periods=1).sum()
data['p2_break_pt_won_count'] = data['p2_break_pt_won'].rolling(window=10, min_periods=1).sum()

# Describe turning points in the data
data['predict'] = 6 - data['point_victor'].shift(-1).fillna(1.5) - data['point_victor'].shift(-2).fillna(1.5) - data['point_victor'].shift(-3).fillna(1.5)
current_swing = 0
data['true swings'] = 0
for index in range(len(data) - 1):
    if index == 0:
        if data['predict'].iloc[index] <= 1:
            current_swing = -1
        elif data['predict'].iloc[index] >= 2:
            current_swing = 1
    else:
        if data['p1_games'].iloc[index] < data['p1_games'].iloc[index + 1]:
            current_swing = 1
        elif data['p2_games'].iloc[index] < data['p2_games'].iloc[index + 1]:
            current_swing = -1
        elif data['p1_sets'].iloc[index] < data['p1_sets'].iloc[index + 1]:
            current_swing = 1
        elif data['p2_sets'].iloc[index] < data['p2_sets'].iloc[index + 1]:
            current_swing = -1
        elif (
            data['p1_points_won'].iloc[index - 1] == data['p1_points_won'].iloc[index]
            and data['p1_points_won'].iloc[index] < data['p1_points_won'].iloc[index + 1]
            and data['p1_points_won'].iloc[index + 1] < data['p1_points_won'].iloc[index + 2]
            and data['p1_points_won'].iloc[index + 2] < data['p1_points_won'].iloc[index + 3]
        ):
            current_swing = 1
        elif (
            data['p2_points_won'].iloc[index - 1] == data['p2_points_won'].iloc[index]
            and data['p2_points_won'].iloc[index] < data['p2_points_won'].iloc[index + 1]
            and data['p2_points_won'].iloc[index + 1] < data['p2_points_won'].iloc[index + 2]
            and data['p2_points_won'].iloc[index + 2] < data['p2_points_won'].iloc[index + 3]
        ):
            current_swing = -1

    data.at[index, 'true swings'] = current_swing
data.at[index + 1, 'true swings'] = current_swing

df = pd.DataFrame(data)

# Split the data into training and testing sets
X = df[['p2_score', 'p1_score', 'is_server', 'p1_recent_points', 'p2_recent_points', 'score_difference_last_5', 'run_difference', 'p1_ace_count', 'p2_ace_count', 'p1_winner_count', 'p2_winner_count', 'p1_unf_err_count', 'p2_unf_err_count', 'p1_break_pt_count', 'p2_break_pt_count', 'p1_break_pt_won_count', 'p2_break_pt_won_count']]

# Use SimpleImputer to handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # Remove the last row, as the last row of 'next_point_victor' is NaN
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_train, X_test, y_train, y_test = train_test_split(X_scaled[:], data['true swings'], test_size=0.2, random_state=42)