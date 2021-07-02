# plotting scores from saved .csv file

import sys
import matplotlib.pyplot as plt
import csv


def plot(csv_path):
    with open(csv_path) as f:
        r = csv.reader(f)
        scores = None
        for row in r:
            scores = [float(s) for s in row]

        # subplots, share axis
        fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
        # plot simple scores
        ax0.plot(range(len(scores)), scores)
        ax0.set_ylabel("Score [Cumulative Reward]")
        ax0.set_xlabel("Episode Number")
        ax0.set_title("Episode Scores")

        # plot average scores
        avg_scores = []
        for i in range(len(scores)):
            avg_scores.append(sum(scores[max(0, i - 99) : i + 1]) / min(i + 1, 100))
        ax1.axhline(y=13.0, xmin=0, xmax=1, color="r", linestyle="--", alpha=0.8)
        ax1.plot(range(len(scores)), avg_scores)
        ax1.set_ylabel("Mean Score [Cumulative Reward]")
        ax1.set_xlabel("Episode Number")
        ax1.set_title("Mean Scores Over Previous 100 Episodes")

        fig.tight_layout()
        plt.show()

        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nERROR:\tinvalid arguments\nUSAGE:\tplot.py <scores.csv path>\n")
    else:
        plot(sys.argv[1])