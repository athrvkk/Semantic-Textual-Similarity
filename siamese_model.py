# File of the Siamese network class
# File: siamese_model.py
# Author: Atharva Kulkarni

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Lambda, Dot
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.regularizers import l2
from tensorflow.keras import losses
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR


import gc
import warnings
warnings.filterwarnings('ignore')



class SiameseModel():


    # ------------------------------------------------------------ Constructor ------------------------------------------------------------
    
    def __init__(self, base_model_type="RoBERTa", activation="relu", kr_rate=0.001, score_loss="mse", cpkt=""):      
        self.kr_rate = kr_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # Set the model activation:
        if activation == "leaky_relu":
            self.activation = LeakyReLU()
            self.kr_initializer = tf.keras.initializers.HeUniform()
        elif activation == "paramaterized_leaky_relu":
            self.activation = PReLU() 
            self.kr_initializer = tf.keras.initializers.HeUniform()          
        elif activation == "relu":
            self.activation = "relu"
            self.kr_initializer = tf.keras.initializers.HeUniform()
        else:
            self.activation = activation
            self.kr_initializer  = tf.keras.initializers.GlorotUniform()

        # Set the regression loss:
        self.score_metric = "mean_squared_error"
        if score_loss == "huber":
            delta = 2.0
            self.score_loss = losses.Huber(delta=delta)
        elif score_loss == "log_cosh":
            self.score_loss = "log_cosh"
        elif score_loss == "mean_squared_logarithmic_error":
            self.score_loss = "mean_squared_logarithmic_error"
        elif score_loss == "mae":
            self.score_loss = "mae"
        else:
            self.score_loss = "mean_squared_error"
        
        # ModelCheckPoint Callback:
        if score_loss == "huber":
            cpkt = cpkt + "-kr-{}-{}-{}-{}".format(self.kr_rate, self.activation, score_loss, delta)
        else:
            cpkt = cpkt + "-kr-{}-{}-{}".format(self.kr_rate, self.activation, score_loss)

        cpkt = cpkt + "-epoch-{epoch:02d}-val-loss-{val_loss:02f}.h5"
        self.model_checkpoint_callback = ModelCheckpoint(filepath=cpkt,
                                                    save_weights_only=True,
                                                    monitor='val_loss',
                                                    mode='auto',
                                                    save_freq = 'epoch',
                                                    save_best_only=True)

        # Reduce Learning Rate on Plateau Callback:
        self.reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', 
                                                    mode='auto',
                                                    factor=0.2, 
                                                    patience=10, 
                                                    min_lr=0.0005, 
                                                    verbose=1)
        # Early Stopping
        self.early_stopping = EarlyStopping(monitor='val_loss', 
                                            patience=20,
                                            verbose=1)
        print("\nActivation: ", self.activation)
        print("Kernel Initializer: ", self.kr_initializer)
        print("Kernel Regularizing Rate: ", self.kr_rate)
        print("\n")





    # ------------------------------------------------------------ Function to prepare input for respective models ------------------------------------------------------------
    def prepare_input(self, utils_obj, df, maxlen=50, padding_type='post', truncating_type='post', mode="train"):
        if mode =="train":
            sent1, sent2, myTokenizer = utils_obj.tokenize_and_pad(df, 
                                                                   maxlen=maxlen,  
                                                                   padding_type=padding_type,
                                                                   truncating_type=truncating_type, 
                                                                   mode="train")
            return sent1, sent2, myTokenizer
        elif mode == "dev" or "test":
            sent1, sent2, _ = utils_obj.tokenize_and_pad(df, 
                                                         maxlen=maxlen, 
                                                         padding_type=padding_type,
                                                         truncating_type=truncating_type, 
                                                         mode="test")
            return sent1, sent2






    def prepare_output(self, df, mode="train"):
        score = df['score'].values.tolist()
        score = np.reshape(score, (len(score), 1))
        if mode == "train":
            score = self.scaler.fit_transform(score)
            score = np.reshape(score, (score.shape[0], 1))
            return score
        elif mode == "dev" or "test":
            score = self.scaler.transform(score)
            score = np.reshape(score, (score.shape[0], 1))
            return score


    # ------------------------------------------------------------ Function to build the model ------------------------------------------------------------
    
    def build(self, embedding_matrix):
        embedding_layer = Embedding(input_dim=embedding_matrix.shape[0], 
                                    output_dim=embedding_matrix.shape[1], 
                                    weights=[embedding_matrix], 
                                    trainable=False)
        bilstm = Bidirectional(LSTM(64, dropout=0.2))
        dense = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))

        input1 = Input(shape=(50,))
        embedding1 = embedding_layer(input1)
        bilstm1 = bilstm(embedding1)
        dense1 = dense(bilstm1)

        input2 = Input(shape=(50,))
        embedding2 = embedding_layer(input2)    
        bilstm2 = bilstm(embedding2)
        dense2 = dense(bilstm2)

        score = Lambda(self.exponent_neg_manhattan_distance, output_shape=self.out_shape)([dense1, dense2])
        
        self.model = Model(inputs=[input1, 
                                   input2], 
                           outputs=score)

        self.model.compile(optimizer = Adam(learning_rate=0.001), 
                           loss=self.score_loss)
        self.model.summary()




    def out_shape(self, shapes):
        return (None, 1)

    def exponent_neg_manhattan_distance(self, vector):
        return K.exp(-K.sum(K.abs(vector[1] - vector[0]), axis=1, keepdims=True))



    # ------------------------------------------------------------ Function to plot model architecture ------------------------------------------------------------
        
    def plot_model_arch(self):
        return plot_model(self.model, show_shapes=True)





    # ------------------------------------------------------------ Function to train the model ------------------------------------------------------------
    
    def train(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=32):
        history = self.model.fit(x_train,
                                 y_train, 
                                 epochs=epochs, 
                                 batch_size=batch_size, 
                                 verbose=1, 
                                 validation_data = (x_val, y_val),
                                 callbacks=[self.model_checkpoint_callback, self.reduce_lr_callback, self.early_stopping])
        return history




    # ------------------------------------------------------------ Function to predict model output ------------------------------------------------------------
    
    def prediction(self, val_essay, model_path=""):
        self.model.load_weights(model_path)
        pred = self.model.predict(val_essay)
        return pred





    # ------------------------------------------------------------ Function to calculate the Pearson's correlation ------------------------------------------------------------
    
    def compute_correlation(self, y_true, y_pred, mode="train", scale=True):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        return pearsonr(y_true, y_pred)
        
        
        


    # ------------------------------------------------------------ Function to calculate the Pearson's correlation ------------------------------------------------------------
    
    def compute_mse(self, y_true, y_pred):  
        return np.average(losses.mean_squared_error(y_true, y_pred))      





    # ------------------------------------------------------------ Function to plot model loss ------------------------------------------------------------
    
    def plot_curves(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'], loc='upper left')
        plt.show() 
