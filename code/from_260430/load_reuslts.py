from calculate_summary_stats import load_summary_stats_single_file

# Load everything
data = load_summary_stats_single_file()

summary_stats = data['summary_stats']  # (500000, 4)
R0 = data['R0']                        # (500000,)
sigma = data['sigma']                  # (500000,)

# Access columns
avg_prev = summary_stats[:, 0]
var_prev = summary_stats[:, 1]
avg_npmi = summary_stats[:, 2]
div_all_isolates = summary_stats[:, 3]
print("summary_stats: ", len(summary_stats))