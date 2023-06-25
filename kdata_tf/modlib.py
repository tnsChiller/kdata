import tensorflow as tf
import numpy as np
keras=tf.keras
kl=keras.layers
K=tf.keras.backend

class OneCycleScheduler(tf.keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)

def Emraldo_mkI():
    opt = keras.optimizers.Adam(learning_rate=1e-5)
    inpt = keras.layers.Input(shape = (33))
    hidden1 = keras.layers.Dense(27, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(inpt)
    hidden2 = keras.layers.Dense(27, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(hidden1)
    hidden3 = keras.layers.Dense(23, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(hidden2)
    hidden4 = keras.layers.Dense(16, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(hidden3)
    hidden5 = keras.layers.Dense(8, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(hidden4)
    hidden6 = keras.layers.Dense(7, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(hidden5)
    out1 = keras.layers.Dense(7, activation = "tanh", kernel_initializer = "lecun_normal")(hidden6)
    model1 = keras.Model(inputs = [inpt], outputs = [out1], name = "Emraldo")
    model1.compile(loss="mean_squared_error",optimizer=opt)
    return model1

def Dolunay_mkI():
    neurons=np.linspace(100,10,8,dtype=np.int32)
    model=keras.models.Sequential()
    layers=[kl.Input(shape=(33))]
            # kl.Dropout(0.1)]
    ctr=0
    for i in range(len(neurons)):
        layers.append(kl.Dense(neurons[i],activation='elu',
                               kernel_initializer='he_normal',))
                               # kernel_constraint=keras.constraints.max_norm(0.8,axis=0),))
                               # kernel_regularizer=keras.regularizers.l1_l2(5e-6,5e-6)))
        # layers.append(kl.BatchNormalization())
        if ctr%1==0:layers.append(kl.Dropout(0.2))
        ctr+=1
    layers.append(kl.Dense(7,activation='tanh'))
    for i in layers:model.add(i)
    opt=keras.optimizers.Nadam()
    
    model.compile(loss='mean_absolute_error',optimizer=opt,metrics=[])
    return model

def Dolunay_dep(neurons=[33,33,20,15,7],
                dropoutPeriod=99,
                dropoutFrequency=0,
                loss="mean_squared_error",
                optimizer="nadam",
                activation="selu",
                kernel="he_normal",
                maxNorm=99,
                l1=0,
                l2=0):
    model=keras.models.Sequential()
    layers=[kl.Input(shape=(33))]
            # kl.Dropout(0.1)]
    ctr=0
    for i in range(len(neurons)):
        layers.append(kl.Dense(neurons[i],activation=activation,
                               kernel_initializer=kernel,
                                kernel_constraint=keras.constraints.max_norm(maxNorm,axis=0),
                                kernel_regularizer=keras.regularizers.l1_l2(l1,l2)))
        if ctr%dropoutPeriod==0:layers.append(kl.Dropout(dropoutFrequency))
        ctr+=1
    layers.append(kl.Dense(7,activation='tanh'))
    for i in layers:model.add(i)
    if optimizer=="adam":
        opt=keras.optimizers.Adam()
    elif optimizer=="nadam":
        opt=keras.optimizers.Nadam()
    elif optimizer=="adamax":
        opt=keras.optimizers.Adamax()
    elif optimizer=="adagrad":
        opt=keras.optimizers.Adagrad()
    elif optimizer=="rmsprop":
        opt=keras.optimizers.RMSprop()
    elif optimizer=="sgd":
        opt=keras.optimizers.SGD()
    else:
        opt=keras.optimizers.SGD()
    
    model.compile(loss='mean_absolute_error',optimizer=opt,metrics=[])
    return model