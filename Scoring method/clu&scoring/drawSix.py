from pandas import *
from keras.engine.saving import load_model
from keras.layers import *
import matplotlib.pyplot as plt
import Keras_AE_application



def draw():

    for j in range(6):
        if j==5:
            encoder = load_model("models/ae0.hdf5")
            encoder_output = encoder.predict(x_train)
        else:
            autoencoder_mse, encoder_output = Keras_AE_application.bulid_ae(x_train, original_dim, 40)

        ALL_minus1 = []
        TTUT = []
        TTTI = []
        TT = []
        TI = []
        ALL_not_minus1 = []

        for i in range(label.shape[0]):
            if label[i] == 0:
                ALL_minus1.append(encoder_output[i, 0:2])
            if label[i] == 1:
                TTTI.append(encoder_output[i, 0:2])
            if label[i] == 2:
                TI.append(encoder_output[i, 0:2])
            if label[i] == 3:
                TTUT.append(encoder_output[i, 0:2])
            if label[i] == 4:
                TT.append(encoder_output[i, 0:2])
            if label[i] == 5:
                ALL_not_minus1.append(encoder_output[i, 0:2])
        ALL_minus1 = np.array(ALL_minus1)
        TTTI = np.array(TTTI)
        TI = np.array(TI)
        TTUT = np.array(TTUT)
        TT = np.array(TT)
        ALL_not_minus1 = np.array(ALL_not_minus1)
        plt.subplot(3, 2, j+1)
        if j==0:
            plt.scatter(TTTI[:, 0], TTTI[:, 1], c='#31A97D', s=3, label='TI=-1,TT=-1,UT≠-1')
            plt.scatter(TI[:, 0], TI[:, 1], c='#3B4D86', s=3, label='TI=-1,TT≠-1,UT≠-1')
            plt.scatter(TTUT[:, 0], TTUT[:, 1], c='#1200FF', s=3, label='TI≠-1,TT=-1,UT=-1')
            plt.scatter(TT[:, 0], TT[:, 1], c='#F7E625', s=3, label='TI≠-1,TT=-1,UT≠-1')
            plt.scatter(ALL_not_minus1[:, 0], ALL_not_minus1[:, 1], c='#006400', s=3, label='TI≠-1,TT≠-1,UT≠-1')
            plt.scatter(ALL_minus1[:, 0], ALL_minus1[:, 1], c='#9F79EE', s=3, label='TI=-1,TT=-1,UT=-1')
            plt.legend(bbox_to_anchor=(0.00, 1.30), loc=2, ncol=3, borderaxespad=0, markerscale=4,
                       fontsize=12)
        else :
            plt.scatter(TTTI[:, 0], TTTI[:, 1], c='#31A97D', s=3)
            plt.scatter(TI[:, 0], TI[:, 1], c='#3B4D86', s=3)
            plt.scatter(TTUT[:, 0], TTUT[:, 1], c='#1200FF', s=3)
            plt.scatter(TT[:, 0], TT[:, 1], c='#F7E625', s=3)
            plt.scatter(ALL_not_minus1[:, 0], ALL_not_minus1[:, 1], c='#006400', s=3)
            plt.scatter(ALL_minus1[:, 0], ALL_minus1[:, 1], c='#9F79EE', s=3)
        title = '('+chr(ord('a') + j)+')'
        plt.xlabel(title,fontsize=14)
        plt.tick_params(labelsize=12)  #设置坐标轴数字大小
        plt.subplots_adjust(wspace=0.2, hspace=0.4)  #调整子图间距
    plt.show()


if __name__ == '__main__':

    dataframe = read_csv('data/addDim51.csv')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    orignal_data = dataset

    label = []

    for i in range(orignal_data.shape[0]):
        if orignal_data[i, 2] == -1 and orignal_data[i, 3] == -1 and orignal_data[i, 4] == -1:
            label.append(0)
        if orignal_data[i, 2] == -1 and orignal_data[i, 3] == -1 and orignal_data[i, 4] != -1:
            label.append(1)
        if orignal_data[i, 2] == -1 and orignal_data[i, 3] != -1 and orignal_data[i, 4] != -1:
            label.append(2)
        if orignal_data[i, 2] != -1 and orignal_data[i, 3] == -1 and orignal_data[i, 4] == -1:
            label.append(3)
        if orignal_data[i, 2] != -1 and orignal_data[i, 3] == -1 and orignal_data[i, 4] != -1:
            label.append(4)
        if orignal_data[i, 2] != -1 and orignal_data[i, 3] != -1 and orignal_data[i, 4] != -1:
            label.append(5)
    label = np.array(label)

    dataframe = read_csv('data/normalization.csv')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    x_train = dataset
    x_train = x_train[:, 1:x_train.shape[1]]
    original_dim = x_train.shape[1]

    draw()
