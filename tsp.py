import mlrose
import numpy as np
from mlrose import TSPGenerator, GARunner

# experiment_name = 'tsp_16'
# problem = TSPGenerator.generate(seed=23, number_of_cities=16, area_height=100, area_width=100)
#
# ga = GARunner(problem=problem,
#               experiment_name=experiment_name,
#               output_directory='~/develop/gatech/ml/random_optimization',
#               seed=23,
#               iteration_list=[1,2,4,8,16,32,64,128,256],
#               max_attempts=2000,
#               population_sizes=[200],
#               mutation_rates=[0.5, 0.6])
#
# # the two data frames will contain the results
# df_run_stats, df_run_curves = ga.run()
#
# exit(1)
#
# experiment_name = 'tsp_32'
# problem = TSPGenerator.generate(seed=23, number_of_cities=32, area_height=100, area_width=100)
#
# ga = GARunner(problem=problem,
#               experiment_name=experiment_name,
#               output_directory='~/develop/gatech/ml/random_optimization',
#               seed=23,
#               iteration_list=[1,2,4,8,16,32,64,128,256, 512, 1024,2048],
#               max_attempts=2000,
#               population_sizes=[200],
#               mutation_rates=[0.5, 0.6])
#
# df_run_stats, df_run_curves = ga.run()


# rhc

#sa
#ga

#mimic


import mlrose
import numpy as np
from mlrose import TSPGenerator, GARunner
from mlrose.opt_probs import DiscreteOpt
from mlrose.runners import RHCRunner, SARunner, GARunner, MIMICRunner


# One Max Problem

SEED = 23
OUTPUT_DIRECTORY = "~/develop/gatech/ml/random_optimization"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


easy_problem = TSPGenerator.generate(seed=23, number_of_cities=8, area_height=100, area_width=100)
complex_problem = TSPGenerator.generate(seed=23, number_of_cities=32, area_height=100, area_width=100)


def plot_curve(df, title='Fitness Curve vs Iterations', row_limit=2000, invert_fitness=True):

    iterations = df['Iteration'][:row_limit]
    fitness = df['Fitness'][:row_limit]
    time = df['Time'][:row_limit]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('fitness', color=color)
    if invert_fitness:
        ax1.plot(iterations, 1/fitness, color=color)
    else:
        ax1.plot(iterations, fitness, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    time_val = df['Time'].values[0]
    ax1.text(0.70, 0.65, 'Time:' + str(round(time_val, 3)), transform=ax1.transAxes, color=color)


  # Create plot
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()




def rhc(easy, hard):
    experiment_name = 'tsp_32'
    problem = hard

    def plot_cohort_curve(cohorts, title='Fitness Curve vs Iterations', row_limit=500, invert_fitness=True):

        colors = ['slategray',
                  'lightsteelblue',
                  'cornflowerblue',
                  'royalblue',
                  'lavender',
                  'navy',
                  'mediumblue',
                  'mediumpurple',
                  'darkviolet',
                  'magenta',
                  'deeppink',
                  'hotpink',
                  'dodgerblue',
                  'darkcyan',
                  'deepskyblue',
                  'lawngreen']

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('iterations')

        for i in range(len(colors)):
            cohort = cohorts[i]
            color = colors[i]

            ax1.set_ylabel('fitness', color=color)

            iterations = cohort['Iteration']
            fitness = cohort['Fitness']
            time = cohort['Time']

            if invert_fitness:
                ax1.plot(iterations, 1/fitness, color=color, label="restart-%s" % (128-16+i,))
            else:
                ax1.plot(iterations, fitness, color=color, label="restart-%s" % (128-16+i,))


        # Create plot
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    rhc = RHCRunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory=OUTPUT_DIRECTORY,
                    seed=SEED,
                    iteration_list=2**np.arange(12),
                    max_attempts=5000,
                    restart_list=[127])
    # the two data frames will contain the results
    df_run_stats, df_run_curves = rhc.run()

    data = pd.read_csv('./' + experiment_name + '/rhc__' + experiment_name + '__curves_df.csv', header=0)

    last_cohort = int(data.shape[0])
    last_set_df = data.tail(n=last_cohort)
    last_set_df = last_set_df.drop(last_set_df.columns[0], axis=1)

    print ("best fitness cohort")
    print(last_set_df.loc[last_set_df['Fitness'].idxmax()])

    cohorts = np.split(last_set_df, 128)

    cohorts = cohorts[:-16]

    plot_cohort_curve(cohorts, 'Fitness Curve Tr Sales - RHC32(restart 127)')

    exit(1)

    experiment_name = 'tsp_8'
    problem = easy

    rhc = RHCRunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory=OUTPUT_DIRECTORY,
                    seed=SEED,
                    iteration_list=2 ** np.arange(7),
                    max_attempts=10000,
                    restart_list=[15],)
    # the two data frames will contain the results
    df_run_stats, df_run_curves = rhc.run()

    data = pd.read_csv('./' + experiment_name + '/rhc__' + experiment_name + '__curves_df.csv', header=0)



    last_cohort = int(data.shape[0])
    last_set_df = data.tail(n=last_cohort)
    last_set_df = last_set_df.drop(last_set_df.columns[0], axis=1)

    cohorts = np.split(last_set_df, 16)

    #  last_set_df = last_set_df.drop(columns=['State'])
    print(last_set_df['Time'])

    plot_cohort_curve(cohorts, 'Fitness Curve OneMax - RHC16(restart 15)', row_limit=1000)

def sa(easy, hard):
    experiment_name = 'tsp_sa_32'
    problem = hard

    sa = SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=5000,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()

    # 50 has a max score 0 566...
    data = pd.read_csv('./' + experiment_name + '/sa__' + experiment_name + '__curves_df.csv', header=0)

    last_cohort = int(data.shape[0])
    last_set_df = data.tail(n=last_cohort)
    last_set_df = last_set_df.drop(last_set_df.columns[0], axis=1)
    plot_curve(last_set_df, 'Fitness Curve TSP - SA32')

    # experiment_name = 'tsp_sa_8'
    # problem = easy
    #
    # sa = SARunner(problem=problem,
    #               experiment_name=experiment_name,
    #               output_directory=OUTPUT_DIRECTORY,
    #               seed=SEED,
    #               iteration_list=2 ** np.arange(14),
    #               max_attempts=5000,
    #               temperature_list=[1, 10, 50, 100])
    # # the two data frames will contain the results
    # df_run_stats, df_run_curves = sa.run()
    #
    # data = pd.read_csv('./' + experiment_name + '/sa__' + experiment_name + '__curves_df.csv', header=0)
    #
    # last_cohort = int(data.shape[0]/4)
    # last_set_df = data.tail(n=last_cohort)
    # last_set_df = last_set_df.drop(last_set_df.columns[0], axis=1)
    #
    # plot_curve(last_set_df, 'Fitness Curve TSP-SA8')

def ga(easy, hard):
    def plot_results(filename, data_set_size, title, row_limit=100):
        data = pd.read_csv(filename, header=0)

        is_55 = data['Mutation Rate'] == 0.55

        m_data = data[is_55]
        plot_curve(m_data, title, row_limit=row_limit)


    def prob_runner(problem, exp_name):
        return GARunner(problem=problem,
              experiment_name=exp_name,
              output_directory='~/develop/gatech/ml/random_optimization',
              seed=23,
              iteration_list=[1,2,4,8,16,32,64,128,256,512, 1024],
              max_attempts=2000,
              population_sizes=[200],
              mutation_rates=[0.55, 0.65])

    def process_problem(prb, name):
        experiment_name = name
        runner = prob_runner(prb, name)
        runner.run()

    def process_experiment(prb, name, graph_title, row_limit=70):
        exp_name = name
        process_problem(prb, exp_name)
        plot_results('./' + exp_name + '/ga__' + exp_name + '__curves_df.csv', 2, graph_title, row_limit=row_limit )


    hard_name = 'tsp_ga_32'
    easy_name = 'tsp_ga_8'
    process_experiment(hard, hard_name, 'Fitness Curve TSP-GA32 (mutation 0.55)')
    # skip the easy problems
    #process_experiment(easy, easy_name, 'Fitness Curve TSP-GA16')
    # 533 @ mutation = 55



def mimic(easy, hard):
        def plot_results(filename, data_set_size, title, row_limit=100):
            data = pd.read_csv(filename, header=0)

            last_cohort = int(data.shape[0] / data_set_size)
            last_set_df = data.tail(n=last_cohort)
            last_set_df = last_set_df.drop(last_set_df.columns[0], axis=1)
            plot_curve(last_set_df, title, row_limit=row_limit)

        def prob_runner(problem, exp_name):
            return MIMICRunner(problem=problem,
                      experiment_name=exp_name,
                      output_directory=OUTPUT_DIRECTORY,
                      seed=SEED,
                      iteration_list=2 ** np.arange(9),
                      max_attempts=700,
                      population_sizes=[900,1800],
                      keep_percent_list=[0.25])


        def process_problem(prb, name):
            runner = prob_runner(prb, name)
            runner.run()

        def process_experiment(prb, name, graph_title, row_limit=50):
            exp_name = name
            process_problem(prb, exp_name)
            plot_results('./' + exp_name + '/mimic__' + exp_name + '__curves_df.csv', 2, graph_title, row_limit=row_limit)

        hard_name = 'tsp_mimic_32'
        easy_name = 'tsp_mimic_8'
        process_experiment(hard, hard_name, 'Fitness Curve TSP-MIMIC32')
       # process_experiment(easy, easy_name, 'Fitness Curve TSP-MIMIC8')


rhc(easy_problem, complex_problem)
sa(easy_problem, complex_problem)
ga(easy_problem, complex_problem)
mimic(easy_problem, complex_problem)

