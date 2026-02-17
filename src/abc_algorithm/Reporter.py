import os

import numpy as np
from decimal import Decimal


class Reporter:
    def __init__(self, abcList):
        self.abcList = abcList
        if abcList[0].conf.PRINT_PARAMETERS:
            self.print_parameters()

        if abcList[0].conf.RUN_INFO:
            self.run_info()
        if abcList[0].conf.SAVE_RESULTS:
            self.save_results()
        if abcList[0].conf.RUN_INFO_COMMANDLINE:
            self.command_line_print()

    def write_header(self):
        if self.save_results:
            with open(self.file_path, 'a') as saveRes:
                saveRes.write(self.header)
                saveRes.close()

    def print_parameters(self):
        for i in range(self.abcList[0].conf.RUN_TIME):
            print(self.abcList[i].experimentID, ". run")
            for j in range(self.abcList[0].conf.DIMENSION):
                print("Global Param[", j + 1, "] ", self.abcList[i].globalParams[j])

    def run_info(self):
        summary = []
        write_text = "%s run: %s Cycle: %s Time: %s"
        for i in range(self.abcList[0].conf.RUN_TIME):
            print_text = write_text % (
            self.abcList[i].experimentID, self.abcList[i].globalOpt, self.abcList[i].cycle, self.abcList[i].globalTime)
            print(print_text)
            summary.append(self.abcList[i].globalOpt)
        print("Mean: ", np.mean(summary), " Std: ", np.std(summary), " Median: ", np.median(summary))

    def command_line_print(self):
        sum = []
        for i in range(self.abcList[0].conf.RUN_TIME):
            sum.append(self.abcList[i].globalOpt)
        print('%1.5E' % Decimal(np.mean(sum)))

    def save_results(self):
        experiment_id = self.abcList[0].experimentID

        if self.abcList[0].conf.OUTPUT_FOLDER_PATH is None:
            experiment_folder = self.abcList[0].conf.EXPERIMENTS_FOLDER_NAME
            if not os.path.exists(experiment_folder):
                os.makedirs(experiment_folder)
            output_folder = "%s/%s" % (experiment_folder, self.abcList[0].conf.OUTPUTS_FOLDER_NAME)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        else:
            output_folder = self.abcList[0].conf.OUTPUT_FOLDER_PATH

        header = "experiment_id, random_seed, seed, problem, number_of_population, max_eval, limit, function, dim, upper_bound, lower_bound, best_validation_accuracy, best_test_accuracy, time \n"
        csvText = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n"

        file_path = "%s/statistics_%s.csv" % (output_folder, experiment_id)
        with open(file_path, 'a') as saveRes:
            is_header = sum(1 for line in open(file_path)) < 1
            if is_header:
                saveRes.write(header)

            for i in range(self.abcList[0].conf.RUN_TIME):
                saveRes.write(csvText.format(
                    self.abcList[i].experimentID,
                    self.abcList[i].conf.RANDOM_SEED,
                    self.abcList[i].conf.SEED,
                    self.abcList[i].conf.SERVICE_NAME,
                    self.abcList[i].conf.NUMBER_OF_POPULATION,
                    self.abcList[i].conf.MAXIMUM_EVALUATION,
                    self.abcList[i].conf.LIMIT,
                    self.abcList[i].conf.OBJECTIVE_FUNCTION.__name__,
                    self.abcList[i].conf.DIMENSION,
                    self.abcList[i].conf.UPPER_BOUND,
                    self.abcList[i].conf.LOWER_BOUND,
                    self.abcList[i].globalOpt,
                    self.abcList[i].best_test_accuracy,
                    self.abcList[i].globalTime,
                ))

            header = "experiment_id,"
            for j in range(self.abcList[0].conf.DIMENSION):

                if j < self.abcList[0].conf.DIMENSION - 1:
                    header = header + "param" + str(j) + ","
                else:
                    header = header + "param" + str(j) + "\n"

            file_path = "%s/params_%s.csv" % (output_folder, experiment_id)

            with open(file_path, 'a') as saveRes:
                is_header = sum(1 for line in open(file_path)) < 1
                if is_header:
                    saveRes.write(header)

                for i in range(self.abcList[0].conf.RUN_TIME):
                    csv_text = str(self.abcList[i].experimentID) + ","
                    for j in range(self.abcList[0].conf.DIMENSION):
                        if j < self.abcList[0].conf.DIMENSION - 1:
                            csv_text = "%s%s," % (csv_text, str(self.abcList[i].globalParams[j]))
                        else:
                            csv_text = "%s%s \n" % (csv_text, str(self.abcList[i].globalParams[j]))
                    saveRes.write(csv_text)
            saveRes.close()

            file_path = "%s/cycle_%s.csv" % (output_folder, experiment_id)
            header = "experiment_id, cycle, fitness_value \n"
            csv_text = "{}, {}, {} \n"

            with open(file_path, 'a') as saveRes:
                is_header = sum(1 for line in open(file_path)) < 1
                if is_header:
                    saveRes.write(header)

                for i in range(self.abcList[0].conf.RUN_TIME):
                    for j in range(self.abcList[i].cycle):
                        saveRes.write(
                            csv_text.format(
                                self.abcList[i].experimentID,
                                str(i),
                                self.abcList[i].globalOpts[j]
                            ))

            # Save results by cycle
            # for i in range(self.abcList[0].conf.RUN_TIME):
            #     folder_path = "%s/%s" % (output_folder, self.abcList[i].conf.RESULT_BY_CYCLE_FOLDER)
            #     if not os.path.exists(folder_path):
            #         os.makedirs(folder_path)
            #     file_path = "%s/%s.txt" % (folder_path, self.abcList[i].experimentID)
            #     with open(file_path, 'a') as saveRes:
            #         for j in range(self.abcList[i].cycle):
            #             saveRes.write(str(self.abcList[i].globalOpts[j]) + "\n")
