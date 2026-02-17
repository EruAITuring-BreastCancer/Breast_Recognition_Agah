import math
import sys
import getopt
import configparser
from deap.benchmarks import sphere, rastrigin, schwefel, rosenbrock, ackley, rand, plane, cigar, h1, himmelblau, \
    schaffer, griewank, rastrigin_scaled, rastrigin_skew, bohachevsky
import os



class Config:

    def __init__(_self, argv):
        config = configparser.ConfigParser()
        config.read(os.path.dirname(os.path.abspath(__file__)) + '/ABC.ini')
        # ####SETTINGS FILE######
        _self.OBJECTIVE_FUNCTION = _self.objFunctionSelector.get(config['DEFAULT']['ObjectiveFunction'], "Error")
        _self.NUMBER_OF_POPULATION = int(config['DEFAULT']['NumberOfPopulation'])
        _self.MAXIMUM_EVALUATION = int(config['DEFAULT']['MaximumEvaluation'])
        _self.LIMIT = int(config['DEFAULT']['Limit'])
        _self.FOOD_NUMBER = int(_self.NUMBER_OF_POPULATION / 2)
        # _self.DIMENSION = int(config['DEFAULT']['Dimension'])
        _self.DIMENSION = 8
        # batch_size, learning_rate, beta_1, beta_2, epoch, l1_reg, l2_reg, dropout
        _self.UPPER_BOUND = [1, 0.002, 0.99999, 0.99999, 1, 0.002, 0.002, 0.5]
        _self.LOWER_BOUND = [0, 0.0005, 0.5, 0.5, 0, 0.0005, 0.0005, 0]
        _self.RUN_TIME = int(config['DEFAULT']['RunTime'])
        _self.SHOW_PROGRESS = bool(config['REPORT']['ShowProgress'] == 'True')
        _self.PRINT_PARAMETERS = bool(config['REPORT']['PrintParameters'] == 'True')
        _self.RUN_INFO = bool(config['REPORT']['RunInfo'] == 'True')
        _self.RUN_INFO_COMMANDLINE = bool(config['REPORT']['CommandLine'] == 'True')
        _self.SAVE_RESULTS = bool(config['REPORT']['SaveResults'] == 'True')
        _self.RESULT_REPORT_FILE_NAME = config['REPORT']['ResultReportFileName']
        _self.PARAMETER_REPORT_FILE_NAME = config['REPORT']['ParameterReportFileName']
        _self.RESULT_BY_CYCLE_FOLDER = config['REPORT']['ResultByCycleFolder']
        _self.EXPERIMENTS_FOLDER_NAME = str(config['REPORT']['ExperimentsFolderName'])
        _self.OUTPUTS_FOLDER_NAME = str(config['REPORT']['OutputsFolderName'])
        _self.RANDOM_SEED = config['SEED']['RandomSeed'] == 'True'
        _self.SEED = int(config['SEED']['Seed'])
        _self.SERVICE_NAME = config['DEFAULT']['ServiceName']
        _self.EXPERIMENT_ID = "Experiment"
        _self.OUTPUT_FOLDER_PATH = None

        # ####SETTINGS FILE######

        # ####SETTINGS ARGUMENTS######
        try:
            opts, args = getopt.getopt(argv, 'hn:m:t:d:l:u:r:o:nrs:sn:s:',
                                       ['help', 'np=', 'max_eval=', 'trial=', 'dim=', 'lower_bound=', 'upper_bound=',
                                        'runtime=',
                                        'obj_fun=', 'output_folder=', 'file_name=', 'param_name=', 'res_cycle_folder=',
                                        'show_functions', 'not_random_seed', 'seed_number=', 'service_name=',
                                        'experiment_id=', 'output_folder_path='])
        except getopt.GetoptError:
            print('Usage: ABCAlgorithm.py -h or --help')
            sys.exit(2)
        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print('-h or --help : Show Usage')
                print('-n or --np : Number of Population')
                print('-m or --max_eval : Maximum Evaluation')
                # print('-t or --trial : Maximum Trial')
                print('-d or --dim : Dimension')
                print('-l or --lower_bound : Lower Bound')
                print('-u or --upper_bound : Upper Bound')
                print('-r or --runtime : Run Time')
                print('-o or --obj_fun : Objective Function')
                print('--show_functions : Show Objective Functions')
                print('--output_folder= [DEFAULT: Outputs]')
                print('--file_name= [DEFAULT: Run_Results.csv]')
                print('--param_name= [DEFAULT: Param_Results.csv]')
                print('--res_cycle_folder= [DEFAULT: ResultByCycle]')

                sys.exit()
            elif opt in ('-n', '--np'):
                _self.NUMBER_OF_POPULATION = int(arg)
            elif opt in ('-m', '--max_eval'):
                _self.MAXIMUM_EVALUATION = int(arg)
            elif opt in ('-d', '--dim'):
                _self.DIMENSION = int(arg)
            elif opt in ('-t', '--trial'):
                _self.LIMIT = int(arg)
            elif opt in ('-l', '--lower_bound'):
                _self.LOWER_BOUND = float(arg)
            elif opt in ('-u', '--upper_bound'):
                _self.UPPER_BOUND = float(arg)
            elif opt in ('-r', '--runtime'):
                _self.RUN_TIME = int(arg)
            elif opt in ('-o', '--obj_fun'):
                _self.OBJECTIVE_FUNCTION = _self.objFunctionSelector.get(arg, "sphere")
            elif opt in ('--output_folder'):
                _self.OUTPUTS_FOLDER_NAME = arg
            elif opt in ('--param_name'):
                _self.PARAMETER_REPORT_FILE_NAME = arg
            elif opt in ('--file_name'):
                _self.RESULT_REPORT_FILE_NAME = arg
            elif opt in ('--res_cycle_folder'):
                _self.RESULT_BY_CYCLE_FOLDER = arg
            elif opt in ('--not_random_seed'):
                _self.RANDOM_SEED = False
            elif opt in ('--seed_number'):
                _self.SEED = int(arg)
            elif opt in ('--service_name'):
                _self.SERVICE_NAME = arg
            elif opt in ('--experiment_id'):
                _self.EXPERIMENT_ID = arg
            elif opt in ('--output_folder_path'):
                _self.OUTPUT_FOLDER_PATH = arg
            elif opt in ('--show_functions'):
                print("We use deap.benchmarks functions. Available functions are listed below:")
                for i in _self.objFunctionSelector:
                    print(i)
                sys.exit()
        # ####SETTINGS ARGUMENTS######

    def setExperimentID(_self, expT):
        _self.EXPERIMENT_ID = expT

    def hyperparameter(individual, service_name):
        def float_decoder(value, number_of_parameters, value_array):
            calc_value = math.floor(number_of_parameters * value)
            if calc_value == number_of_parameters:
                calc_value -= 1
            return value_array[calc_value]

        # 0 -> batch_size
        # 1 -> learning_rate
        # 2 -> beta_1
        # 3 -> beta_2
        # 4 -> epoch
        # 5 -> l1_regularization
        # 6 -> l2_regularization
        # 7 -> dropout_rate
        batch_size = float_decoder(individual[0], 5, [4, 8, 16, 32, 64])
        epoch_size = float_decoder(individual[4], 5, [20, 40, 60, 80, 100])
        learning_rate = round(individual[1], 5)
        beta_1 = round(individual[2], 5)
        beta_2 = round(individual[3], 5)
        l1_reg = round(individual[5], 5)
        l2_reg = round(individual[6], 5)
        dropout = round(individual[7], 5)
        print("batch_size: ", batch_size, "learning_rate: ", learning_rate, "beta_1: ", beta_1, "beta_2: ", beta_2,
              "epoch_size: ", epoch_size, "l1_reg: ", l1_reg, "l2_reg: ", l2_reg, "dropout: ", dropout)

        """ret_val = get_fitness(batch_size=batch_size, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                              epochs=epoch_size, l1_reg=l1_reg,
                              l2_reg=l2_reg, dropout=dropout, s=service_name)
"""
#        return ret_val

        return 0
    # ######FUNCTION_LIST######
    objFunctionSelector = {
        'sphere': sphere,
        'rastrigin': rastrigin,
        'rosenbrock': rosenbrock,
        'rand': rand,
        'plane': plane,
        'cigar': cigar,
        'h1': h1,
        'ackley': ackley,
        'bohachevsky': bohachevsky,
        'griewank': griewank,
        'rastrigin_scaled': rastrigin_scaled,
        'rastrigin_skew': rastrigin_skew,
        'schaffer': schaffer,
        'schwefel': schwefel,
        'himmelblau': himmelblau,
        'hyperparameter': hyperparameter
    }
    # ######FUNCTION_LIST######
