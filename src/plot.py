# plotting scores from saved .csv file

import sys
import matplotlib.pyplot as plt
import csv


def plot(agt_scores_csv_path, avg_scores_csv_path):
    agt_scores, avg_scores = [], []
    with open(agt_scores_csv_path) as f:
        r = csv.reader(f)
        for row in r:
            agt_scores.append([float(s) for s in row])
    with open(avg_scores_csv_path) as f:
        r = csv.reader(f)
        for row in r:
            avg_scores = [float(s) for s in row]
    n = len(avg_scores)
    r = range(n)
    ticks = [*range(1, n, n // 5), n + 1]

    # subplots, share axis
    fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
    
    # plot simple agent scores and their mean
    for s in agt_scores:
        ax0.plot(r, s, alpha=0.5)
    ax0.plot(r, avg_scores, label="mean agent score", color="k")
    ax0.set_xticks(ticks)
    ax0.legend(loc="lower right")
    ax0.set_ylabel("Score [Cumulative Episode Reward]")
    ax0.set_xlabel("Episode Number")
    ax0.set_title("Agent Episode Scores")

    # plot mean of mean agent scores over the past 100 episodes
    avg_avg_scores = []
    for i in r:
        avg_avg_scores.append(sum(avg_scores[max(0, i - 99) : i + 1]) / min(i + 1, 100))
    ax1.axhline(y=30.0, xmin=0, xmax=1, color="r", linestyle="--", alpha=0.5)
    ax1.plot(r, avg_avg_scores)
    ax1.set_xticks(ticks)
    ax1.set_ylabel("Mean Score [Cumulative Episode Reward]")
    ax1.set_xlabel("Episode Number")
    ax1.set_title("Mean of Mean Agent Scores Over Previous 100 Episodes")

    fig.tight_layout()
    plt.show()

        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("\nERROR:\tinvalid arguments\nUSAGE:\tplot.py <scores.csv path> <scores_avg.csv path>\n")
    else:
        plot(sys.argv[1], sys.argv[2])