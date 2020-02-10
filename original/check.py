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
    for i in range(check_data.shape[0]):
        for ii in range(i):
            if (check_data[ii, 12] <= check_data[i, 12]) and check_data[ii, 13] <= check_data[i, 13] and check_data[ii,
                 14] <= check_data[i, 14] and check_data[ii, 15] <= check_data[i, 15] and check_data[ii,
                 16] <= check_data[i, 16] and check_data[ii, 17] <= check_data[i, 17]:
                print(check_data[i, 2], check_data[ii, 2])
                err_.append((check_data[i, 2], check_data[ii, 2]))
                break
    save = DataFrame(err_)
    save.columns = [name, name + "2"]
    save = save.sort_values(by=[name+"2", name])
    save.to_csv('errors/errors__'+name, index=False, header=True)


check("s2s", 'score_3.csv')
