from tensorflow import keras
from matplotlib import pyplot
import kdata_tf_lib as ktl
import numpy as np
import time

def test_mod():
    opt = keras.optimizers.Adam(learning_rate=1e-5)
    inpt = keras.layers.Input(shape = (33,))
    hidden1 = keras.layers.Dense(27, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(inpt)
    hidden2 = keras.layers.Dense(27, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(hidden1)
    hidden3 = keras.layers.Dense(23, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(hidden2)
    hidden4 = keras.layers.Dense(16, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(hidden3)
    hidden5 = keras.layers.Dense(8, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(hidden4)
    hidden6 = keras.layers.Dense(7, use_bias=True, bias_initializer="zeros", activation = "selu", kernel_initializer = "lecun_normal")(hidden5)
    out1 = keras.layers.Dense(7, activation = "tanh", kernel_initializer = "lecun_normal")(hidden6)
    mod = keras.Model(inputs = [inpt], outputs = [out1], name = "Emraldo")
    mod.compile(loss="mean_squared_error",optimizer=opt)
    return mod

def train_model(mod, num_games, mod_name, epochs=100, batch_size=64, iterations=1):
    for i in range(iterations):
        t0 = time.perf_counter()
        x,y = ktl.generate_data(num_games, mod)
        train_hist = mod.fit(x,y,
                             shuffle=True,
                             epochs=epochs,
                             batch_size=batch_size,
                             verbose=1)
        t1 = time.perf_counter()
        print(f"Training machine: iteration {i+1}/{iterations} complete. ({t1-t0:1.2}s)")
    mod.save(f"machines/{mod_name}.h5")
        
def play_games(num, mods, mod_names):
    players=[]
    for i in range(6):
        idx = i % len(mods)
        players.append(mods[idx])
        idx += 1
    gs = ktl.gameState(min(num, 100000), players)
    t0 = time.perf_counter()
    gs.gamEval()
    t1 = time.perf_counter()
    print(f"{num} games played, time = {t1 - t0}")
    stacks = gs.stacks
    len_modsin=len(mods)
    if len_modsin == 2:
        m_pos = [[0, 2, 4], [1, 3, 5]]
    elif len_modsin == 3:
        m_pos = [[0, 3], [1, 4], [2, 5]]
    else:
        m_pos = [[0], [1], [2], [3], [4], [5]]
    gains=[]
    for m in m_pos:
        m_gains = [stacks[:,idx] for idx in m]
        gains.append(sum(m_gains))
    gains = np.array(gains)
    tot_gains = np.zeros(gains.shape)
    tot_gains[:,0] = gains[:,0]
    for i in range(1,tot_gains.shape[1]):
        tot_gains[:,i] = tot_gains[:,i-1] + gains[:,i]
       

    pyplot.figure(figsize=(15,10))
    for i in range(tot_gains.shape[0]):
        pyplot.plot(tot_gains[i],)
    pyplot.legend(mod_names)
    pyplot.xlabel("Game Number")
    pyplot.ylabel("Total Stack")
    pyplot.savefig(f"pics/test_game.png")
    return tot_gains[0,-1] - tot_gains[1,-1]