# github page: https://abcolony.github.io/
# github repository: https://github.com/abcolony/ABCPython


import datetime
import sys
import time

import ABC
import Config
from Reporter import Reporter


#from workspace.root.data import get_train_test_df
#from workspace.root.hyp_optimizer import Config, ABC
#from workspace.root.hyp_optimizer.Reporter import Reporter
#from workspace.root.mlp.utils import make_folders, make_optimization_folder
#from workspace.root.utils.parameters import SERVICES
#import cProfile

# @profile


def ABCAlgorithm(argv):

    abc_conf = Config.Config(argv)
    abc_list = list()
    # expT = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "").replace(":", "")

#    make_folders()
    #    try:
    #    new_save_path = make_optimization_folder()
    #except FileExistsError:
    #    time.sleep(20)
    #   new_save_path = make_optimization_folder()

    # for s in SERVICES:
    s = abc_conf.SERVICE_NAME
    #train_df, test_df = get_train_test_df(s, test_frac=0.2)
    #train_df.to_csv(new_save_path + s + '_train.csv', index=False)
    #test_df.to_csv(new_save_path + s + '_test.csv', index=False)

    for run in range(abc_conf.RUN_TIME):

        abc = ABC.ABC(abc_conf)
        start_time = time.time() * 1000
        abc.initial()
        abc.memorize_best_source()
        while not(abc.stopping_condition()):
            print("EvalCount: ", abc.evalCount, "Best Model: ", abc.best_model,
                  "Best Fitness: ", abc.globalOpt)
            abc.send_employed_bees()
            abc.calculate_probabilities()
            abc.send_onlooker_bees()
            abc.memorize_best_source()
            abc.send_scout_bees()
            abc.increase_cycle()
            print("EvalCount: ", abc.evalCount, "Best Model: ", abc.best_model,
                  "Best Fitness: ", abc.globalOpt)

        abc.globalTime = time.time() * 1000 - start_time
        abc_list.append(abc)
    Reporter(abc_list)


# if __name__ == '__main__':
#     # cProfile.run("main(sys.argv[1:])", 'restats')
#     main(sys.argv[1:])
#
