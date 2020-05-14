from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import math
from sklearn.preprocessing import MinMaxScaler


def normalization_T(a, b, neg, T, flag):
    '''
    
    :param a: 分段下界
    :param b: 分段上界
    :param neg: -1和MinMaxScalar范围的最小距离
    :param T: 传入待归一化矩阵T，shape = (None,1)
    :param flag:正反向标记
    :return:归一化后的矩阵
    '''
    if neg != 0:
        neg_index = np.argwhere(T == -1)
        # print(neg_index)
        T[neg_index] = 100000
        a_index = np.argwhere(T < a)
        T[neg_index] = -1
        b_index = np.argwhere(T > b)
        temp1 = (T >= a)
        temp2 = (T <= b)
        temp = temp1 * temp2
        max = np.max(T[temp])
        T[neg_index] = max
        T[a_index] = max
        T[b_index] = max
        T = np.reshape(T, (-1, 1))
        T = T * (flag)
        s = MinMaxScaler(feature_range=(neg, 1))
        T = s.fit_transform(T)
        if flag <= 0:
            T[a_index] = 1
            T[neg_index] = 0
            T[b_index] = neg
        else:
            T[a_index] = neg
            T[neg_index] = 0
            T[b_index] = 1
        return T
    else:
        a_index = np.argwhere(T < a)
        b_index = np.argwhere(T > b)
        neg_index = np.argwhere(T == -1)
        # print(neg_index)
        T[neg_index] = b
        T[a_index] = a
        T[b_index] = b
        T = np.reshape(T, (-1, 1))
        T = T * (flag)
        s = MinMaxScaler(feature_range=(0, 1))
        T = s.fit_transform(T)
        print(T)
        return T


def normalization_TI(TI):
    '''
   
    :param TI: 待归一化矩阵TI
    :return: 归一化矩阵TI
    '''
    index2 = np.argwhere(TI == -1)
    index1 = np.argwhere(TI != -1)
    TI[index1] = 1
    # print(index1)
    # print(index2)
    TI[index2] = 0
    return TI


def normalization_other(a, b, P, flag):
    ''' 
    :param a: 分段下界
    :param b: 分段上界
    :param P: 传入待归一化矩阵P，shape=(None,1)
    :param flag: 为正代表数值越大越好，为负代表数值越小越好
    :return: 归一化后的P
    '''
    a_index = np.argwhere(P < a)
    b_index = np.argwhere(P > b)
    # print(P[a_index])
    # print(P[b_index])
    temp1 = (P >= a)
    temp2 = (P <= b)
    temp = temp1 * temp2
    max = np.max(P[temp])
    P[a_index] = max
    P[b_index] = max
    P = np.reshape(P, (-1, 1))
    P = P * flag
    s = MinMaxScaler(feature_range=(0, 1))
    P = s.fit_transform(P)
    if flag < 0:
        P[a_index] = 0
        P[b_index] = 1
    else:
        P[a_index] = 1
        P[b_index] = 0
    return P


def normalization(dim=8, TT_a=None, TT_b=None, TT_neg=0.3,
                  UT_a=None, UT_b=None, UT_neg=0.3,
                  LMS_a=None, LMS_b=None,
                  CML_a=None, CML_b=None,
                  MLR_a=None, MLR_b=None,
                  SP_a=None, SP_b=None,
                  RP_a=None, RP_b=None):
    dataframe = read_csv('3dataoutput.csv')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    dataset = np.array(dataset)
    if TT_a == None:
        if TT_neg == 0:
            TT_a = -1
        else:
            TT_a = np.min(dataset[:, 3][np.argwhere(dataset[:, 3] != -1)])
    if TT_b == None:
        TT_b = np.max(dataset[:, 3][np.argwhere(dataset[:, 3] != -1)])
    if UT_a == None:
        if UT_neg == 0:
            UT_a = -1
        else:
            UT_a = np.min(dataset[:, 4][np.argwhere(dataset[:, 4] != -1)])
    if UT_b == None:
        UT_b = np.max(dataset[:, 4][np.argwhere(dataset[:, 4] != -1)])
    if LMS_a == None:
        LMS_a = np.min(dataset[:, 5][np.argwhere(dataset[:, 5] != -1)])
    if LMS_b == None:
        LMS_b = np.max(dataset[:, 5][np.argwhere(dataset[:, 5] != -1)])
    if CML_a == None:
        CML_a = np.min(dataset[:, 6][np.argwhere(dataset[:, 6] != -1)])
    if CML_b == None:
        CML_b = np.max(dataset[:, 6][np.argwhere(dataset[:, 6] != -1)])
    if MLR_a == None:
        MLR_a = np.min(dataset[:, 7][np.argwhere(dataset[:, 7] != -1)])
    if MLR_b == None:
        MLR_b = np.max(dataset[:, 7][np.argwhere(dataset[:, 7] != -1)])
    if SP_a == None:
        SP_a = np.min(dataset[:, 0])
    if SP_b == None:
        SP_b = np.max(dataset[:, 0])
    if RP_a == None:
        RP_a = np.min(dataset[:, 1])
    if RP_b == None:
        RP_b = np.max(dataset[:, 1])
    SP = normalization_other(SP_a, SP_b, dataset[:, 0], 1)
    RP = normalization_other(RP_a, RP_b, dataset[:, 1], 1)
    TI = normalization_TI(dataset[:, 2])
    TT = normalization_T(TT_a, TT_b, TT_neg, dataset[:, 3], -1)
    UT = normalization_T(UT_a, UT_b, UT_neg, dataset[:, 4], -1)
    LMS = normalization_other(LMS_a, LMS_b, dataset[:, 5], -1)
    CML = normalization_other(CML_a, CML_b, dataset[:, 6], -1)
    MLR = normalization_other(MLR_a, MLR_b, dataset[:, 7], -1)
    dataset[:, 0] = SP.reshape(-1)
    dataset[:, 1] = RP.reshape(-1)
    dataset[:, 2] = TI.reshape(-1)
    dataset[:, 3] = TT.reshape(-1)
    dataset[:, 4] = UT.reshape(-1)
    dataset[:, 5] = LMS.reshape(-1)
    dataset[:, 6] = CML.reshape(-1)
    dataset[:, 7] = MLR.reshape(-1)
    dataset = dataset[:, (dim - 8):]
    df = DataFrame(dataset)
    df.columns = ['SP', 'RP', 'TI', 'TT', 'UT', 'LMS', 'CML', 'MLR']
    df.to_csv('3n1.csv', index=False, header=True)
    return dataset


normalization(UT_neg=0, TT_neg=0)
