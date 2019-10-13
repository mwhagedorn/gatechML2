# rhc

#sa
#ga

#mimic


import mlrose
import numpy as np
from mlrose import FlipFlopGenerator
from mlrose.opt_probs import DiscreteOpt
from mlrose.runners import RHCRunner, SARunner, GARunner, MIMICRunner


# One Max Problem

SEED = 23
OUTPUT_DIRECTORY = "~/develop/gatech/ml/random_optimization"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



easy_problem = FlipFlopGenerator.generate(seed=SEED, size=16)
complex_problem = FlipFlopGenerator.generate(seed=SEED, size=64)

def plot_curve(df, title='Fitness Curve vs Iterations', row_limit=2000):

    iterations = df['Iteration'][:row_limit]
    fitness = df['Fitness'][:row_limit]
    time = df['Time'][:row_limit]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('fitness', color=color)
    ax1.plot(iterations, fitness, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    time_val = df['Time'].values[0]
    ax1.text(0.70, 0.65, 'Time:' + str(round(time_val, 3)), transform=ax1.transAxes, color=color)

  # Create plot
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


easy_problem = FlipFlopGenerator.generate(seed=SEED, size=16)
complex_problem = FlipFlopGenerator.generate(seed=SEED, size=64)

def rhc(easy, hard):
    experiment_name = 'ff_64'
    problem = hard

    def plot_curve_rhc(df, title, short):
        iterations = df['Iteration']
        fitness1 = df['Fitness_10']
        if short:
            fitness2 = df['Fitness_48']
        else:
            fitness2 = df['Fitness_127']

        fig, ax1 = plt.subplots()

        time_10 = df['Time_10'].values[0]
        if short:
            time_48 = df['Time_48'].values[0]
        else:
            time_127 = df['Time_127'].values[0]


        color = 'tab:red'
        color2 = 'tab:blue'
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('fitness', color=color)
        ax1.plot(iterations, fitness1, color=color, label='fitness restart=10')
        if short:
            ax1.plot(iterations, fitness2, color=color2, label='fitness restart=48')
        else:
            ax1.plot(iterations, fitness2, color=color2, label='fitness restart=127')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.text(0.65, 0.8, 'Time (restart=10):' + str(round(time_10, 3)), transform=ax1.transAxes, color=color)
        if short:
            ax1.text(0.65, 0.85, 'Time (restart=48):' + str(round(time_48, 3)), transform=ax1.transAxes, color=color2)
        else:
            ax1.text(0.65, 0.85, 'Time (restart=127):' + str(round(time_127, 3)), transform=ax1.transAxes, color=color2)

        # Create plot
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_results(filename, data_set_size, title):
        data = pd.read_csv(filename, header=0)

        is_10 = data['current_restart'] == 10
        is_127 = data['current_restart'] == 127
        is_48 = data['current_restart'] == 47

        r2_data = data[is_10]

        r4_data = data[is_127]
        short = False
        if r4_data.empty:
            short = True
            r4_data = data[is_48]

        if short:
            r2_data['Fitness_48'] = r4_data['Fitness'].values
            r2_data['Time_48'] = r4_data['Time'].values
        else:
            r2_data['Fitness_127'] = r4_data['Fitness'].values
            r2_data['Time_127'] = r4_data['Time'].values

        r2_data.rename(columns={'Fitness': 'Fitness_10', 'Time': 'Time_10'}, inplace=True)
        plot_curve_rhc(r2_data, title, short)

    def prob_runner(problem, exp_name):
        return  RHCRunner(problem=problem,
                    experiment_name=exp_name,
                    output_directory=OUTPUT_DIRECTORY,
                    seed=SEED,
                    iteration_list=2 ** np.arange(10),
                    max_attempts=5000,
                    restart_list=[127])

    def process_problem(prb, name):
        runner = prob_runner(prb, name)
        runner.run()

    def process_experiment(prb, name, graph_title):
        exp_name = name
        process_problem(prb, exp_name)
        plot_results('./' + exp_name + '/rhc__' + exp_name + '__curves_df.csv', 2, graph_title)

    hard_name = 'ff_64'
    easy_name = 'ff_16'
    process_experiment(hard, hard_name, 'Fitness Curve FF-RHC64')
    process_experiment(easy, easy_name, 'Fitness Curve FF-RHC16')

    # the two data frames will contain the results


def sa(easy, hard):
    def plot_curve_sa(df, title='Fitness Curve vs Iterations', row_limit=11000):
        iterations = df['Iteration'][0:row_limit]
        fitness1 = df['Fitness_1'][0:row_limit]
        fitness50 = df['Fitness_100'][0:row_limit]

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        color2 = 'tab:blue'
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('fitness', color=color)
        ax1.plot(iterations, fitness1, color=color, label='fitness T=1')
        ax1.plot(iterations, fitness50, color=color2, label='fitness T=100')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.text(0.75, 0.6, 'Time (T=1):' + str(round(df['Time_1'][0], 3)), transform=ax1.transAxes, color=color)
        ax1.text(0.75, 0.65, 'Time (T=100):' + str(round(df['Time_100'][0], 3)), transform=ax1.transAxes, color=color2)

        # Create plot
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    experiment_name = 'ff_sa_64'
    problem = hard

    sa = SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=5000,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    # the two data frames will contain the results
    #df_run_stats, df_run_curves = sa.run()

    # data = pd.read_csv('./' + experiment_name + '/sa__' + experiment_name + '__curves_df.csv', header=0)
    #
    # is_1 = data['Temperature'] == 1
    # is_50 = data['Temperature'] == 100
    #
    # t1_data = data[is_1][0:2662]
    # t50_data = data[is_50][0:2662]
    #
    # f50 = t50_data['Fitness']
    # t1_data['Fitness_100'] = f50.values
    # t1_data['Time_100'] = t50_data['Time'].values

    #t1_data.rename(columns={'Fitness': 'Fitness_1', 'Time': 'Time_1'}, inplace=True)
    #
    #plot_curve_sa(t1_data, 'Fitness Curve OneMax - SA64', row_limit=1000)

    experiment_name = 'ff_sa_16'
    problem = easy

    sa = SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=5000,
                  temperature_list=[1, 10, 50, 100])
    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()

    data = pd.read_csv('./' + experiment_name + '/sa__' + experiment_name + '__curves_df.csv', header=0)

    is_1 = data['Temperature'] == 1
    is_50 = data['Temperature'] == 50

    t1_data = data[is_1][0:455]
    t50_data = data[is_50][0:455]

    f50 = t50_data['Fitness']
    t1_data['Fitness_50'] = f50.values
    t1_data['Time_50'] = t50_data['Time'].values

    t1_data.rename(columns={'Fitness': 'Fitness_1', 'Time': 'Time_1'}, inplace=True)
    #
    plot_curve_sa(t1_data, 'Fitness Curve OneMax - SA16', row_limit=1000)

def ga(easy, hard):
    def plot_results(filename, data_set_size, title, row_limit=100):
        data = pd.read_csv(filename, header=0)

        is_2 = data['Population Size'] == 200
        is_8 = data['Population Size'] == 800

        p2_data = data[is_2]
        p8_data = data[is_8]

        plot_curve(p2_data,'FF: pop 200 mutation 0.65', row_limit=row_limit)
        plot_curve(p8_data, 'FF: pop 800 mutation 0.65', row_limit=row_limit)


    def prob_runner(problem, exp_name):
        return GARunner(problem=problem,
              experiment_name=exp_name,
              output_directory='~/develop/gatech/ml/random_optimization',
              seed=23,
              iteration_list=[1,2,4,8,16,32,64,128,256,512],
              max_attempts=2000,
              population_sizes=[200,800],
              mutation_rates=[0.65])

    def process_problem(prb, name):
        experiment_name = name
        runner = prob_runner(prb, name)
        runner.run()

    def process_experiment(prb, name, graph_title, row_limit=2662):
        exp_name = name
        process_problem(prb, exp_name)
        plot_results('./' + exp_name + '/ga__' + exp_name + '__curves_df.csv', 2, graph_title, row_limit=row_limit )


    hard_name = 'ff_ga_64'
    easy_name = 'ff_ga_16'
    process_experiment(hard, hard_name, 'Fitness Curve FF-GA64')
    #process_experiment(easy, easy_name, 'Fitness Curve FF-GA16')




def mimic(easy, hard):
        def plot_results(filename, data_set_size, title, row_limit=100):
            data = pd.read_csv(filename, header=0)

            is_2 = data['Population Size'] == 450
            is_8 = data['Population Size'] == 900

            p2_data = data[is_2]
            p8_data = data[is_8]

            plot_curve(p2_data, 'Mimic: pop 450 keep pct 0.25', row_limit=row_limit)
            plot_curve(p8_data, 'Mimic: pop 900 mutation 0.25', row_limit=row_limit)

        def prob_runner(problem, exp_name):
            return MIMICRunner(problem=problem,
                      experiment_name=exp_name,
                      output_directory=OUTPUT_DIRECTORY,
                      seed=SEED,
                      iteration_list=2 ** np.arange(7),
                      max_attempts=700,
                      population_sizes=[450,900],
                      keep_percent_list=[0.25])


        def process_problem(prb, name):
            runner = prob_runner(prb, name)
            runner.run()

        def process_experiment(prb, name, graph_title, row_limit=50):
            exp_name = name
            process_problem(prb, exp_name)
            plot_results('./' + exp_name + '/mimic__' + exp_name + '__curves_df.csv', 2, graph_title, row_limit=row_limit)

        hard_name = 'ff_mimic_64'
        easy_name = 'ff_mimic_16'
        process_experiment(hard, hard_name, 'Fitness Curve OneMax-MIMIC64')
        #process_experiment(easy, easy_name, 'Fitness Curve OneMax-MIMIC16')


rhc(easy_problem, complex_problem)
sa(easy_problem, complex_problem)
ga(easy_problem, complex_problem)
mimic(easy_problem, complex_problem)

