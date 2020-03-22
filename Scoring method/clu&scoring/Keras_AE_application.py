from keras.models import Model
from keras.layers import *
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.cluster import DBSCAN

batch_size = 256
encoding_dim = 2

def best_DB_para(x_train):
    es = [0.1,0.2,0.3,0.4,0.5]
    ms = 10
    current_epchos = 50
    mse,input_data = bulid_ae(x_train,epochs = current_epchos)
    for i in range(5):
        db = DBSCAN(eps=es[i], min_samples=10).fit(input_data)
        plt.subplot(2, 3, i + 1)
        plt.scatter(input_data[:, 0], input_data[:, 1], c=db.labels_,s = 5)
        plt.title("epchos = %d , eps = %.2f , ms = %d "% (current_epchos,es[i],ms))
    plt.show()
def bulid_ae(x_train,original_dim,epochs = 40):
    input = Input(shape=(original_dim,))
    encoded = Dense(16, activation='relu')(input)
    encoded = Dense(4, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)
    decoded = Dense(4, activation='relu')(encoder_output)
    decoded = Dense(16, activation='relu')(decoded)
    decoded = Dense(original_dim, activation='tanh')(decoded)

    autoencoder = Model(inputs=input, outputs=decoded)

    encoder = Model(inputs=input, outputs=encoder_output)

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size , verbose=2)
    autoencoder_output = autoencoder.predict(x_train)
    autoencoder_mse = mean_squared_error(x_train, autoencoder_output)
    encoded_imgs = encoder.predict(x_train)
    return autoencoder_mse,encoded_imgs










