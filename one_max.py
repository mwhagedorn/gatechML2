# rhc

#sa
#ga

#mimic


import mlrose
import numpy as np
from mlrose.opt_probs import DiscreteOpt
from mlrose.runners import RHCRunner, SARunner, GARunner, MIMICRunner

class OneMaxGenerator:
    @staticmethod
    def generate(seed, size=20):
        np.random.seed(seed)
        problem = DiscreteOpt(length=size, fitness_fn=mlrose.fitness.OneMax())
        return problem

# One Max Problem

SEED = 23
OUTPUT_DIRECTORY = "~/develop/gatech/ml/random_optimization"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_curve(df, title='Fitness Curve vs Iterations'):
    iterations = df['Iteration']
    fitness = df['Fitness']
    time = df['Time']

    avg_fit = int(fitness.median())

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('fitness', color=color)
    ax1.plot(iterations, fitness, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.text(0.7,0.6, 'Time:'+str(round(df['Time'][0],4)), transform=ax1.transAxes)

    # Create plot
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


easy_problem = OneMaxGenerator.generate(seed=SEED, size=16)
complex_problem = OneMaxGenerator.generate(seed=SEED, size=64)

def rhc(easy, hard):
    experiment_name = 'one_max_rhc_64'
    problem = hard

    rhc = RHCRunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory=OUTPUT_DIRECTORY,
                    seed=SEED,
                    iteration_list=2 ** np.arange(10),
                    max_attempts=5000,
                    restart_list=[30])
    # the two data frames will contain the results
   #df_run_stats, df_run_curves = rhc.run()

    data = pd.read_csv('./' + experiment_name + '/rhc__one_max_rhc_64__curves_df.csv', header=0)

    last_set_df = data.head(n=512)

    #  last_set_df = last_set_df.drop(columns=['State'])
    print(last_set_df['Time'])

    plot_curve(last_set_df, 'Fitness Curve OneMax - RHC64(restart 30)')

    experiment_name = 'one_max_rhc_16'
    problem = easy

    rhc = RHCRunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory=OUTPUT_DIRECTORY,
                    seed=SEED,
                    iteration_list=2 ** np.arange(10),
                    max_attempts=5000,
                    restart_list=[30])
    # the two data frames will contain the results
    df_run_stats, df_run_curves = rhc.run()

    data = pd.read_csv('./' + experiment_name + '/rhc__one_max_rhc_16__curves_df.csv', header=0)

    last_set_df = data.head(n=512)

    #  last_set_df = last_set_df.drop(columns=['State'])
    print(last_set_df['Time'])

    plot_curve(last_set_df, 'Fitness Curve OneMax - RHC16(restart 30)')

def sa(easy, hard):
    def plot_curve_sa(df, title='Fitness Curve vs Iterations', row_limit=11000):
        iterations = df['Iteration'][0:row_limit]
        fitness1 =  df['Fitness_1'][0:row_limit]
        fitness50 = df['Fitness_50'][0:row_limit]

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        color2 = 'tab:blue'
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('fitness', color=color)
        ax1.plot(iterations, fitness1, color=color, label='fitness T=1')
        ax1.plot(iterations, fitness50, color=color2, label='fitness T=50')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.text(0.75, 0.6, 'Time (T=1):' + str(round(df['Time_1'][0], 3)), transform=ax1.transAxes, color=color)
        ax1.text(0.75, 0.65, 'Time (T=50):' + str(round(df['Time_50'][0], 3)), transform=ax1.transAxes, color=color2)

        # Create plot
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    experiment_name = 'one_max_sa_64'
    problem = hard

    sa = SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=5000,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    # the two data frames will contain the results
    # df_run_stats, df_run_curves = sa.run()

    data1 = pd.read_csv('./' + experiment_name + '/sa__one_max_sa_64__curves_df.csv', header=0)
    is_1 = data1['Temperature'] == 1
    is_50 = data1['Temperature'] == 50

    t1_data = data1[is_1]
    t50_data = data1[is_50]

    f50 = t50_data['Fitness']
    t1_data['Fitness_50'] = f50.values
    t1_data['Time_50'] = t50_data['Time'].values

    t1_data.rename(columns={'Fitness':'Fitness_1', 'Time':'Time_1'}, inplace=True)
    #
    plot_curve_sa(t1_data, 'Fitness Curve OneMax - SA64', row_limit=1000)


    experiment_name = 'one_max_sa_16'
    problem = easy

    sa = SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=5000,
                  temperature_list=[1, 10, 50, 100])
    # the two data frames will contain the results
    #df_run_stats, df_run_curves = sa.run()

    data1 = pd.read_csv('./' + experiment_name + '/sa__one_max_sa_16__curves_df_t1.csv', header=0)
    it_col = data1['Iteration']
    time_col = data1['Time']
    fitness1 = data1['Fitness']
    frame = { 'Iteration': it_col, 'Time_1': time_col, 'Fitness_1':fitness1}
    the_data = pd.DataFrame(frame)

    data50 = pd.read_csv('./' + experiment_name + '/sa__one_max_sa_16__curves_df_t50.csv', header=0)
    time_col = data50['Time']
    fitness50 = data50['Fitness']
    the_data['Fitness_50'] = fitness50
    the_data['Time_50'] = time_col

    plot_curve_sa(the_data, 'Fitness Curve OneMax - SA16')
    exit(1)

def ga(easy, hard):
    def plot_curve_ga(df, title='Fitness Curve vs Iterations', row_limit=11000):
        iterations = df['Iteration'][0:row_limit]
        fitness_1 = df['Fitness_0.5'][0:row_limit]
        fitness_2 = df['Fitness_0.6'][0:row_limit]

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        color2 = 'tab:blue'
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('fitness', color=color)
        ax1.plot(iterations, fitness_1, color=color, label='fitness mutation=0.5')
        ax1.plot(iterations, fitness_2, color=color2, label='fitness mutation=0.6')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.text(0.70, 0.6, 'Time (m=0.5):' + str(round(df['Time_0.5'][0], 3)), transform=ax1.transAxes, color=color)
        ax1.text(0.70, 0.65, 'Time (m=0.6):' + str(round(df['Time_0.6'][0], 3)), transform=ax1.transAxes, color=color2)

        # Create plot
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_results(filename, data_set_size, title, row_limit=100):
        data1 = pd.read_csv(filename, header=0)
        is_5 = data1['Mutation Rate'] == 0.5
        is_6 = data1['Mutation Rate'] == 0.6

        r5_data = data1[is_5][0:5000]
        r6_data = data1[is_6][0:5000]

        r5_data['Fitness_0.6'] = r6_data['Fitness'].values
        r5_data['Time_0.6'] = r6_data['Time'].values

        r5_data.rename(columns={'Fitness': 'Fitness_0.5', 'Time': 'Time_0.5'}, inplace=True)
        #
        plot_curve_ga(r5_data, title, row_limit=row_limit)


    def prob_runner(problem, exp_name):
        return GARunner(problem=problem,
              experiment_name=exp_name,
              output_directory='~/develop/gatech/ml/random_optimization',
              seed=23,
              iteration_list=[1,2,4,8,16,32,64,128,256],
              max_attempts=2000,
              population_sizes=[200],
              mutation_rates=[0.5, 0.6])

    def process_problem(prb, name):
        experiment_name = name
        runner = prob_runner(prb, name)
        runner.run()

    def process_experiment(prb, name, graph_title, row_limit=1000):
        exp_name = name
        process_problem(prb, exp_name)
        plot_results('./' + exp_name + '/ga__' + exp_name + '__curves_df.csv', 2, graph_title, row_limit=row_limit )


    hard_name = 'one_max_ga_64'
    easy_name = 'one_max_ga_16'
    process_experiment(hard, hard_name, 'Fitness Curve OneMax-GA64', row_limit=50)
    process_experiment(easy, easy_name, 'Fitness Curve OneMax-GA16', row_limit=50)




def mimic(easy, hard):
    def plot_curve_mim(df, title='Fitness Curve vs Iterations', row_limit=11000):
        iterations = df['Iteration'][0:row_limit]
        fitness_1 = df['Fitness_200'][0:row_limit]
        fitness_2 = df['Fitness_400'][0:row_limit]

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        color2 = 'tab:blue'
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('fitness', color=color)
        ax1.plot(iterations, fitness_1, color=color, label='fitness population=200')
        ax1.plot(iterations, fitness_2, color=color2, label='fitness population=400')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.text(0.70, 0.6, 'Time (p=200):' + str(round(df['Time_200'][0], 3)), transform=ax1.transAxes, color=color)
        ax1.text(0.70, 0.65, 'Time (p=400):' + str(round(df['Time_400'][0], 3)), transform=ax1.transAxes, color=color2)

        # Create plot
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_results(filename, data_set_size, title, row_limit=100):
        data = pd.read_csv(filename, header=0)

        is_200 = data['Population Size'] == 200
        is_400 = data['Population Size'] == 400

        r2_data = data[is_200][0:5000]
        r4_data = data[is_400][0:5000]

        r2_data['Fitness_400'] = r4_data['Fitness'].values
        r2_data['Time_400'] = r4_data['Time'].values

        r2_data.rename(columns={'Fitness': 'Fitness_200', 'Time': 'Time_200'}, inplace=True)
        plot_curve_mim(r2_data, title, row_limit=row_limit)

    def prob_runner(problem, exp_name):
        return MIMICRunner(problem=problem,
                  experiment_name=exp_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(6),
                  max_attempts=700,
                  population_sizes=[200,400],
                  keep_percent_list=[0.2])


    def process_problem(prb, name):
        runner = prob_runner(prb, name)
        runner.run()

    def process_experiment(prb, name, graph_title, row_limit=50):
        exp_name = name
        #process_problem(prb, exp_name)
        plot_results('./' + exp_name + '/mimic__' + exp_name + '__curves_df.csv', 2, graph_title, row_limit=row_limit)

    hard_name = 'one_max_mimic_64'
    easy_name = 'one_max_mimic_16'
    process_experiment(hard, hard_name, 'Fitness Curve OneMax-Mimic64')
    process_experiment(easy, easy_name, 'Fitness Curve OneMax-Mimic16')


rhc(easy_problem, complex_problem)
sa(easy_problem, complex_problem)
ga(easy_problem, complex_problem)
mimic(easy_problem, complex_problem)

