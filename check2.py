import pandas as pd
import numpy as np
from pandas import DataFrame


def check(name, path):
    data = pd.read_csv(path)
    # data = data.values
    data = DataFrame(data)
    check_data = data.sort_values(by=name, ascending=True)
    check_data = check_data.values
    err_ = []
    flag = 0
    for i in range(check_data.shape[0]):
        for ii in range(i):
            if check_data[ii, 2] > check_data[i, 2]:
                print(2)
                if check_data[ii, 3] > check_data[i, 3]:
                    print(3)
                    if check_data[ii, 4] > check_data[i, 4]:
                        print(4)
                        if check_data[ii, 5] > check_data[i, 5]:
                            print(5)
                            if check_data[ii, 6] > check_data[i, 6]:
                                print(6)
                                if check_data[ii, 7] > check_data[i, 7]:
                                    print(7)
                                    if check_data[ii, 8] > check_data[i,
                                8] and check_data[ii, 9] > check_data[i, 9] and check_data[ii, 10] > check_data[i, 10] and check_data[ii,
                                11] > check_data[i, 11] and check_data[ii,12] > check_data[i, 12] and check_data[ii, 13] > check_data[i,
                                13]:
                                        print(check_data[i, 2], check_data[ii, 2])
                                        err_.append(check_data[i, :])
                                        err_.append(check_data[ii, :])
                                        flag = 1
                # err_.append()
    if flag:
        save = DataFrame(err_)
        save.columns = ['SP', 'RP', 'TI', 'TT', 'UT', 'LMS', 'CML', 'MLR', 'RP2', 'TI2', 'TT2', 'UT2', 'LMS2', 'CML2', 'MLR2',
                        'mean', 'trans', 'pre_trans', 's2s', 'pres2s', 'dae']
        # save = save.sort_values(by=[name+"2", name])
        save.to_csv('errors/errors__' + name + '.csv', index=False, header=True)
    else:
        print(name + 'No Errors!')


for name in ['mean', 'trans', 'pre_trans', 's2s', 'pres2s', 'dae']:
    print(name + 'checking...')
    check(name, 'scores/scores_s911.csv')
