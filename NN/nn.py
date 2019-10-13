import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mlrose import NeuralNetwork, GeomDecay
from sklearn.metrics import accuracy_score
import pickle as pk

import matplotlib.pyplot as plt




def dump_df_to_disk(df, runner_name, df_name, experiment_name, output_directory):

    def build_data_filename(output_directory, runner_name, experiment_name, df_name, ext=''):
        # ensure directory exists
        try:
            os.makedirs(os.path.join(output_directory, experiment_name))
        except:
            pass

        # return full filename
        if len(ext) > 0 and not ext[0] == '.':
            ext = f'.{ext}'
        return os.path.join(output_directory,
                            experiment_name,
                            f'{runner_name.lower()}__{experiment_name}__{df_name}{ext}')

    filename_root = build_data_filename(output_directory=output_directory,
                                        runner_name=runner_name,
                                        experiment_name=experiment_name,
                                        df_name=df_name)

    pk.dump(df, open(f'{filename_root}.p', "wb"))
    df.to_csv(f'{filename_root}.csv')



data = pd.read_csv('sonar.all-data',header=None)
# https://www.simonwenkel.com/2018/08/23/revisiting_ml_sonar_mines_vs_rocks.html


# identify sonar column names
data.columns = ['X0','X1','X2','X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
                'X10', 'X11','X12','X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19',
                'X20','X21','X22','X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29',
                'X30','X31','X32','X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39',
                'X40','X41','X42','X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49',
                'X50','X51','X52','X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'Class']

data['Class'] = np.where(data['Class'] == 'R',0,1) #Change the Class representation

# shuffle the data rows
data = data.reindex(np.random.RandomState(seed=42).permutation(data.index))

X = data.drop('Class',axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

SEED = 23

# 70 hidden units in a single layer, learning rate=0.15, regularization(L2)=0.4 had an accuracy of 0.78.



def rhc():
    def plot_curve_rhc(df, title):
        iterations = df['Iteration']
        fitness1 = df['Fitness']
        time_val = df['Time'].values[0]
        train_acc = df['TrainAcc'].values[0]
        test_acc = df['TestAcc'].values[0]

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        color2 = 'tab:blue'
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('fitness', color=color)
        ax1.plot(iterations, 1/fitness1, color=color, label='fitness')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.text(0.65, 0.7, 'Time:' + str(round(time_val, 3)), transform=ax1.transAxes, color=color)
        ax1.text(0.65, 0.65, 'Train Acc:' + str(round(train_acc, 3)), transform=ax1.transAxes, color=color)
        ax1.text(0.65, 0.55, 'Test Acc:' + str(round(test_acc, 3)), transform=ax1.transAxes, color=color)
        # Create plot
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def run_experiment(experiment_name, model, restarts=130, max_iters=2000, analyze=True, title='Foo'):
        model.restarts = restarts
        model.max_iters = max_iters
        run_start = time.perf_counter()
        if analyze:
            result = model.fit(X_train, y_train)
            run_end = time.perf_counter()
            print(f'Run time: {run_end - run_start}')
            run_time = run_end - run_start
            curves_df = pd.DataFrame(columns=['Fitness', 'Time', 'Restarts'])
            curves_df['Fitness'] = model.fitness_curve
            curves_df['Time'] = run_time
            curves_df['Restarts'] = restarts
            curves_df['Iteration'] = curves_df.index

            # Predict labels for train set and assess accuracy
            y_train_pred = model.predict(X_train)

            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            print("RHC NN train accuracy")
            print(y_train_accuracy)
            curves_df['TrainAcc'] = y_train_accuracy

            # Predict labels for test set and assess accuracy
            y_test_pred = model.predict(X_test)

            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            curves_df['TestAcc'] = y_test_accuracy
            print("RHC NN test accuracy")
            print(y_test_accuracy)

            dump_df_to_disk(curves_df, 'rhc', 'curves_df', experiment_name, 'rhc')

        filename = 'rhc/' + experiment_name +'/rhc__'+ experiment_name  +'__curves_df.csv'
        data = pd.read_csv(filename, header=0)
        plot_curve_rhc(data, title)

    experiment_name = 'nn_rhc'

    restarts = 1
    model = NeuralNetwork(hidden_nodes=[70], activation='relu', \
                              algorithm='random_hill_climb', max_iters=2000, \
                              bias=True, is_classifier=True, learning_rate=0.15, \
                              early_stopping=True, clip_max=5, max_attempts=100, \
                              random_state=SEED,
                              restarts=restarts,
                              curve=True)

    #run_experiment('nn_rhc_r30', model, 30)
    run_experiment('nn_rhc_r50', model, 50, analyze=False, title='Fitness Curve NN(70) - restarts 50')
    #run_experiment('nn_rhc_r130', model, 130)
    run_experiment('nn_rhc_r130_it3000', model, restarts=130, max_iters = 3000, analyze=False, title='Fitness Curve NN(70) - restarts 130')


def sa():
    def plot_curve_sa(df, title):
        iterations = df['Iteration']
        fitness1 = df['Fitness']
        time_val = df['Time'].values[0]
        train_acc = df['TrainAcc'].values[0]
        test_acc = df['TestAcc'].values[0]

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        color2 = 'tab:blue'
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('fitness', color=color)
        ax1.plot(iterations, 1 / fitness1, color=color, label='fitness')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.text(0.65, 0.35, 'Time:' + str(round(time_val, 3)), transform=ax1.transAxes, color=color)
        ax1.text(0.65, 0.30, 'Train Acc:' + str(round(train_acc, 3)), transform=ax1.transAxes, color=color)
        ax1.text(0.65, 0.25, 'Test Acc:' + str(round(test_acc, 3)), transform=ax1.transAxes, color=color)
        # Create plot
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def run_experiment(experiment_name, model, init_temp=15.0, max_it = 2000, title="Foo", analyze=True):
        model.schedule = GeomDecay(init_temp=init_temp)
        model.max_iters = max_it

        run_start = time.perf_counter()
        if analyze:
            result = model.fit(X_train, y_train)
            run_end = time.perf_counter()
            print(f'Run time: {run_end - run_start}')
            run_time = run_end - run_start
            curves_df = pd.DataFrame(columns=['Fitness', 'Time', 'Restarts'])
            curves_df['Fitness'] = model.fitness_curve
            curves_df['Time'] = run_time
            curves_df['Restarts'] = restarts
            curves_df['Iteration'] = curves_df.index

            # Predict labels for train set and assess accuracy
            y_train_pred = model.predict(X_train)

            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            print("RHC NN train accuracy")
            print(y_train_accuracy)
            curves_df['TrainAcc'] = y_train_accuracy

            # Predict labels for test set and assess accuracy
            y_test_pred = model.predict(X_test)

            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            curves_df['TestAcc'] = y_test_accuracy
            print("RHC NN test accuracy")
            print(y_test_accuracy)

            dump_df_to_disk(curves_df, 'sa', 'curves_df', experiment_name, 'sa')

        filename = 'sa/' + experiment_name + '/sa__' + experiment_name + '__curves_df.csv'
        data = pd.read_csv(filename, header=0)
        plot_curve_sa(data, title)


    experiment_name = 'nn_sa'

    restarts = 1
    model = NeuralNetwork(hidden_nodes=[70], activation='relu', \
                          algorithm='simulated_annealing', max_iters=2000, \
                          bias=True, is_classifier=True, learning_rate=0.15, \
                          early_stopping=True, clip_max=5, max_attempts=100, \
                          random_state=SEED,
                          schedule=GeomDecay(init_temp=15.0),
                          curve=True)

    #run_experiment('nn_sa_t15', model, 15)
    #run_experiment('nn_sa_t15', model, 30)
    run_experiment('nn_sa_t50', model, 50, title='Fitness Curve NN(70) - SA T=50', analyze=False)
    #run_experiment('nn_rhc_r80', model, 80)
    #run_experiment('nn_rhc_r130_it4000', model, 130, 4000)
    run_experiment('nn_rhc_r130_it5000', model, 130, 5000,title='Fitness Curve NN(70) - SA T=130', analyze=False )

def ga():
    def plot_curve_ga(df, title):
        iterations = df['Iteration']
        fitness1 = df['Fitness']
        time_val = df['Time'].values[0]
        train_acc = df['TrainAcc'].values[0]
        test_acc = df['TestAcc'].values[0]

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        color2 = 'tab:blue'
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('fitness', color=color)
        ax1.plot(iterations, 1 / fitness1, color=color, label='fitness')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.text(0.65, 0.35, 'Time:' + str(round(time_val, 3)), transform=ax1.transAxes, color=color)
        ax1.text(0.65, 0.30, 'Train Acc:' + str(round(train_acc, 3)), transform=ax1.transAxes, color=color)
        ax1.text(0.65, 0.25, 'Test Acc:' + str(round(test_acc, 3)), transform=ax1.transAxes, color=color)
        # Create plot
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def run_experiment(experiment_name, model, mutation_pct=0.2, population=200, analyze=True, title="foo"):
        model.mutation_prob = mutation_pct
        model.pop_size = population
        if analyze:
            run_start = time.perf_counter()
            result = model.fit(X_train, y_train)
            run_end = time.perf_counter()
            print(f'Run time: {run_end - run_start}')
            run_time = run_end - run_start
            curves_df = pd.DataFrame(columns=['Fitness', 'Time', 'Restarts'])
            curves_df['Fitness'] = model.fitness_curve
            curves_df['Time'] = run_time
            curves_df['Restarts'] = restarts
            curves_df['Iteration'] = curves_df.index

            # Predict labels for train set and assess accuracy
            y_train_pred = model.predict(X_train)

            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            print("RHC NN train accuracy")
            print(y_train_accuracy)
            curves_df['TrainAcc'] = y_train_accuracy

            # Predict labels for test set and assess accuracy
            y_test_pred = model.predict(X_test)

            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            curves_df['TestAcc'] = y_test_accuracy
            print("RHC NN test accuracy")
            print(y_test_accuracy)

            dump_df_to_disk(curves_df, 'ga', 'curves_df', experiment_name, 'ga')

        filename = 'ga/' + experiment_name + '/ga__' + experiment_name + '__curves_df.csv'

        data = pd.read_csv(filename, header=0)
        plot_curve_ga(data, title)

    experiment_name = 'nn_ga'

    restarts = 1
    model = NeuralNetwork(hidden_nodes=[70], activation='relu', \
                          algorithm='genetic_alg', max_iters=2000, \
                          bias=True, is_classifier=True, learning_rate=0.15, \
                          early_stopping=True, clip_max=5, max_attempts=100, \
                          random_state=SEED,
                          pop_size=200,
                          mutation_prob=0.55,
                          curve=True)

    #run_experiment('nn_ga_m05', model, 0.5)
    #run_experiment('nn_ga_m05_p400', model, 0.5, 400)
    #run_experiment('nn_ga_m05_p800', model, 0.5, 800)
    #run_experiment('nn_ga_m025', model, 0.25, 200)
    run_experiment('nn_ga_m01', model,0.1,200, title="NN(70)- GA, pop = 200, mutation=0.1", analyze=False)
    # run_experiment('nn_rhc_r80', model, 80)
    # run_experiment('nn_rhc_r130_it4000', model, 130, 4000)
    # run_experiment('nn_rhc_r130_it5000', model, 130, 5000)


rhc()
sa()
ga()