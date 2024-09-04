import subprocess
import numpy as np
import csv

from hyperopt import hp, tpe, Trials, fmin
from hyperopt import STATUS_OK
from timeit import default_timer as timer

import os
iter = 1

def objective(args):
    # define args
    global iter
    str_list = []
    for str in args:
        str_list.append('--{}={}'.format(str,args[str]))
        globals()[str] = args[str]
    print('[iter = {}]'.format(iter), str_list)
    iter += 1

    # train
    start = timer()
    # 先根据数据名称获取相应的配置

    # 构建命令行参数
    python_executable = '/root/miniconda3/envs/deepst_env/bin/python'
    command = [python_executable, 'analyze_DLPFC.py', *str_list]

    # 运行命令
    output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end = timer()

    time_train = end - start

    test_output = output.stdout.decode('utf-8')

    # handle output
    lines = test_output.split('\n')
    # value = lines[-2].split(': ')[0]
    # metrics = lines[-3].split()
    # ari = float(metrics[3])
    metrics = lines[2].split()
    ari = float(metrics[5])

    # objective
    loss = ari
    results = {'loss': -loss, 'status': STATUS_OK, 'epoch': args['epoch'], 'lr': args['lr'],'weight_decay': args['weight_decay'],
               'lambda1': args['lambda1'],'lambda2': args['lambda2'], 'lambda3': args['lambda3'],
                'ari': ari,  'time': time_train}

    # save
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow(results.values())
    of_connection.close()
    return results

# space = {'learning_rate': hp.loguniform('learning_rate_AE', np.log(0.00005),np.log(0.001)),
#          'a': hp.quniform('a', 0, 1, 0.001)}
space = {'epoch': hp.quniform('epoch', 50, 500, 50),
         # 'lr': hp.choice('lr', [0.000001,0.000003,0.000004,0.000005,0.000008,0.00001,0.00003,0.00004,0.00005,0.00008,
         #                        0.0001,0.0003,0.0004,0.0005,0.0008,0.001,0.003,0.004,0.005,0.008]),
         'lr': hp.choice('lr', [0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000008,
                                0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00008]),
         'weight_decay': hp.choice('weight_decay', [0.00001,0.00003,0.00005,0.00008,0.0001,0.0003,0.0005,0.0008,0.001,0.003,0.005,0.008]),
         'lambda1': hp.choice('lambda1', [0.001,0.003,0.005,0.008,0.01,0.03,0.05,0.08]),
         'lambda2': hp.choice('lambda2', [1.0,2.0,3.0,4.0,5.0]),
         'lambda3': hp.choice('lambda3', [1.0,2.0,3.0,4.0,5.0,8.0,10.0]),
         }

tpe_algo = tpe.suggest
trials = Trials()

# File to save first results
out_file = '151673_429.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
# Write the headers to the file
writer.writerow(['loss','status', 'epoch', 'lr', 'weight_decay', 'lambda1','lambda2', 'lambda3', 'ari',  'time'])
of_connection.close()

best = fmin(fn=objective, space=space, algo=tpe_algo, trials=trials, max_evals=300, rstate=np.random.RandomState(3))

# os.system("/usr/bin/shutdown")