import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('ga__example_experiment__run_stats_df.csv',header=0)

half = int(data.shape[0]/2)
last_set_df = data.tail(n=half)
last_set_df = last_set_df.drop(last_set_df.columns[0], axis=1)

last_set_df = last_set_df.drop(columns=['State'])
print(last_set_df['Time'])


def plot_curve(df, title='Fitness Curve vs Iterations'):

    iterations = df['Iterations']
    fitness = df['Fitness']
    time = df['Time']

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('fitness (1/d)', color=color)
    ax1.plot(iterations, 1/fitness, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('time (s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(iterations, time, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

  # Create plot
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


plot_curve(last_set_df, 'Fitness Curve GA, mutation % 0.6')