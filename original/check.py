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
            if (check_data[ii, 12] > check_data[i, 12]) and check_data[ii, 13] > check_data[i, 13] and check_data[ii,
                 14] > check_data[i, 14] and check_data[ii, 15] > check_data[i, 15] and check_data[ii,
                 16] > check_data[i, 16] and check_data[ii, 17] > check_data[i, 17]:
                print(check_data[i, 2], check_data[ii, 2])
                err_.append(check_data[i, :])
                err_.append(check_data[ii, :])
                flag = 1
                # err_.append()
    if flag :

        save = DataFrame(err_)
        save.columns = ['x', 'y', 'SP', 'RP', 'TI', 'TT', 'UT', 'LMS', 'CML', 'MLR', 'n_SP',
                        'n_RP', 'n_TI', 'n_TT', 'n_UT', 'n_LMS', 'n_CML', 'n_MLR', 'Label',
                        'clu&sco', 'mean', 'c&s_0_1', 'vae', 's2s', 'pres2s', 'dae']
        # save = save.sort_values(by=[name+"2", name])
        save.to_csv('errors/errors__'+name+'.csv', index=False, header=True)
    else:
        print(name+'No Errors!')


# check("dae", 'scores_55.csv')
