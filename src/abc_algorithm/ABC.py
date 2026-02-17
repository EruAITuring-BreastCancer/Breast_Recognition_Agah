__author__ = "Omur Sahin"

import sys
import numpy as np
from deap.benchmarks import random

#from workspace.root.data import get_train_test_df
#from workspace.root.mlp.utils import make_optimization_folder, make_folders


class ABC:

    def __init__(self, conf):
        self.conf = conf
        self.foods = np.zeros((self.conf.FOOD_NUMBER, self.conf.DIMENSION))
        self.f = np.ones(self.conf.FOOD_NUMBER)
        self.test_accuracy = np.ones(self.conf.FOOD_NUMBER)
        self.fitness = np.ones(self.conf.FOOD_NUMBER) * np.iinfo(int).max
        self.trial = np.zeros(self.conf.FOOD_NUMBER)
        self.prob = [0 for x in range(self.conf.FOOD_NUMBER)]
        self.globalParams = [0 for x in range(self.conf.DIMENSION)]
        self.globalTime = 0
        self.evalCount = 0
        self.cycle = 0
        self.experimentID = conf.EXPERIMENT_ID
        self.globalOpts = list()
        self.globalOpt = np.iinfo(int).max
        self.model_names = [str() for x in range(self.conf.FOOD_NUMBER)]
        self.best_model = ""
        self.best_test_accuracy = None

        if not (conf.RANDOM_SEED):
            random.seed(conf.SEED)

    def calculate_function(self, sol):
        try:
            return self.conf.OBJECTIVE_FUNCTION(sol, self.conf.SERVICE_NAME)

        except ValueError as err:
            print(
                "An exception occured: Upper and Lower Bounds might be wrong. (" + str(err) + " in calculate_function)")
            sys.exit()

    def calculate_fitness(self, fun):
        self.increase_eval()
        if fun >= 0:
            result = 1 / (fun + 1)
        else:
            result = 1 + abs(fun)
        return result

    def increase_eval(self):
        print("Progress: " + str(self.evalCount) + "/" + str(self.conf.MAXIMUM_EVALUATION))
        self.evalCount += 1

    def stopping_condition(self):
        if self.globalOpt == 0:
            return True
        status = bool(self.evalCount >= self.conf.MAXIMUM_EVALUATION)
        return status

    def memorize_best_source(self):
        for i in range(self.conf.FOOD_NUMBER):
            if self.f[i] < self.globalOpt:
                self.globalOpt = np.copy(self.f[i])
                self.globalParams = np.copy(self.foods[i][:])
                self.best_model = self.model_names[i]
                self.best_test_accuracy = self.test_accuracy[i]

    def init(self, index):
        if not (self.stopping_condition()):
            for i in range(self.conf.DIMENSION):
                self.foods[index][i] = random.random() * (self.conf.UPPER_BOUND[i] - self.conf.LOWER_BOUND[i]) + \
                                       self.conf.LOWER_BOUND[i]
            solution = np.copy(self.foods[index][:])
            calc_val = self.calculate_function(solution)
            self.f[index] = 1 - calc_val['valid_accuracy']
            self.test_accuracy[index] = 1 - calc_val['test_accuracy']
            self.model_names[index] = calc_val['model_name']
            self.fitness[index] = self.calculate_fitness(self.f[index])
            self.trial[index] = 0

    def initial(self):
        for i in range(self.conf.FOOD_NUMBER):
            self.init(i)
            self.increase_eval()
        self.globalOpt = np.copy(self.f[0])
        self.globalParams = np.copy(self.foods[0][:])

    def calculate_neighbour_solution(self, change_index):
        param2change = random.randint(0, self.conf.DIMENSION - 1)
        neighbour = random.randint(0, self.conf.FOOD_NUMBER - 1)
        while neighbour == change_index:
            neighbour = random.randint(0, self.conf.FOOD_NUMBER - 1)

        solution = np.copy(self.foods[change_index][:])
        r = random.random()
        solution[param2change] = round(self.foods[change_index][param2change] + (
                    self.foods[change_index][param2change] - self.foods[neighbour][param2change]) * (r - 0.5) * 2)
        if solution[param2change] < self.conf.LOWER_BOUND[param2change]:
            solution[param2change] = self.conf.LOWER_BOUND[param2change]
        if solution[param2change] > self.conf.UPPER_BOUND[param2change]:
            solution[param2change] = self.conf.UPPER_BOUND[param2change]
        return solution

    def send_employed_bees(self):
        i = 0
        while (i < self.conf.FOOD_NUMBER) and (not (self.stopping_condition())):
            solution = self.calculate_neighbour_solution(i)
            calc_val = self.calculate_function(solution)
            obj_val = 1 - calc_val['valid_accuracy']
            fitness_sol = self.calculate_fitness(obj_val)
            if fitness_sol > self.fitness[i]:
                self.trial[i] = 0
                self.foods[i][:] = np.copy(solution)
                self.f[i] = obj_val
                self.test_accuracy[i] = calc_val['test_accuracy']
                self.fitness[i] = fitness_sol
                self.model_names[i] = calc_val['model_name']
            else:
                self.trial[i] = self.trial[i] + 1
            i += 1

    def calculate_probabilities(self):
        max_fit = np.copy(max(self.fitness))
        for i in range(self.conf.FOOD_NUMBER):
            self.prob[i] = (0.9 * (self.fitness[i] / max_fit)) + 0.1

    def send_onlooker_bees(self):
        i = 0
        t = 0
        while (t < self.conf.FOOD_NUMBER) and (not (self.stopping_condition())):
            r = random.random()
            if r < self.prob[i]:
                t += 1
                solution = self.calculate_neighbour_solution(i)
                calc_val = self.calculate_function(solution)
                obj_val = 1 - calc_val['valid_accuracy']
                fitness_sol = self.calculate_fitness(obj_val)
                if fitness_sol > self.fitness[i]:
                    self.trial[i] = 0
                    self.foods[i][:] = np.copy(solution)
                    self.f[i] = obj_val
                    self.test_accuracy[i] = calc_val['test_accuracy']
                    self.fitness[i] = fitness_sol
                    self.model_names[i] = calc_val['model_name']
                else:
                    self.trial[i] = self.trial[i] + 1
            i = (i + 1) % self.conf.FOOD_NUMBER

    def send_scout_bees(self):
        if np.amax(self.trial) >= self.conf.LIMIT:
            self.init(self.trial.argmax(axis=0))

    def increase_cycle(self):
        self.globalOpts.append(self.globalOpt)
        self.cycle += 1

    def setExperimentID(self, run, t):
        self.experimentID = t + "-" + str(run)
