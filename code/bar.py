import matplotlib.pyplot as plt
import numpy as np

# Learned Weights Set 1
weights_set1 = [-0.37300154, 0.23546694, 0.20590442, 0.45111854, -0.45111854, -0.01278271,
                   -0.31297638, 0.09944217, -0.07898622, 0.22368248, -0.19597448, 0.02491982,
                   0.01116199, -0.07211369, -0.30019459, 0.45508234, -0.13018146]





# Learned Weights Set 2
weights_set2 = [-0.41331256, 0.13360196, 0.34012693, 0.41347448, -0.41347448, -0.03600107,
                -0.14310323, -0.02224884, -0.03772534, 0.20868659, 0.07922033, -0.20974141,
                0.02902037, -0.04252096, -0.06850752, 0.46634935, -0.37609134]

# Feature names
feature_names = ['p1_score', 'p2_score', 'is_server', 'p1_recent_points', 'p2_recent_points',
                 'score_difference_last_5', 'run_difference', 'p1_ace_count', 'p2_ace_count',
                 'p1_winner_count', 'p2_winner_count', 'p1_unf_err_count', 'p2_unf_err_count',
                 'p1_break_pt_count', 'p2_break_pt_count', 'p1_break_pt_won_count', 'p2_break_pt_won_count']

# Generate bar plot
plt.figure(figsize=(12, 8))
bar_width = 0.35
bar_positions_set1 = np.arange(len(feature_names))
bar_positions_set2 = bar_positions_set1 + bar_width

plt.barh(bar_positions_set1, weights_set1, height=bar_width, label='Weights of Alcaraz', color='lightcoral')
plt.barh(bar_positions_set2, weights_set2, height=bar_width, label='Weights of Djokovic', color='skyblue')

# Add numeric values to the bars
for i, value in enumerate(weights_set1):
    plt.text(value, i, f'{value:.3f}', ha='center', va='center', color='black', fontsize=14)

for i, value in enumerate(weights_set2):
    plt.text(value, i + bar_width, f'{value:.3f}', ha='center', va='center', color='black', fontsize=12)

plt.xlabel('Weight Value', fontsize=14)
# plt.title('Comparison of Learned Weights between Set 1 and Set 2')
plt.yticks(bar_positions_set1 + bar_width / 2, feature_names, fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Adjust the position of the plot
plt.subplots_adjust(left=0.2)

plt.show()
