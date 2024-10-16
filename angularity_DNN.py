from ModelsContainer import ModelsContainer
from energyflow.archs.dnn import DNN
from energyflow.archs.moment import EFN_moment, PFN_moment
from numpy import average
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, to_categorical
import toploader
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))



# Parameters 
train = 1000000
val = 50000
test = 50000
k_order = int(sys.argv[1]) 
run_name = sys.argv[3]
dataset = sys.argv[2] # either "qg" or "top"


epochs = 150
batch_size=512
callbacks =None
verbose = 2


num_models_to_train = 5 ##number of models to use to make error bars
order_list = [k_order,] #
input_dim = 2
output_dim = 1


# Directory Handling
base_dir =  "/n/home01/rikab/MomentAnalysis/Data"
run_dir = os.path.join(base_dir, run_name)
run_dir = os.path.join(run_dir, f"order_{k_order}")
model_dir = os.path.join(run_dir, "Models")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
    os.makedirs(model_dir, exist_ok=True)
topdir = "/n/holyscratch01/iaifi_lab/rikab/top"

###########



max_L = 128
F_width = 100


betas = np.linspace(0, 5, 31)

Ls = []
Ls = [2,]

num_samples = len(betas)


# max_L_per_order = [2**(8-k_order),]
# F_min, F_max = 100
# Phi_min, Phi_max = 100
# logN_max = 6.5


# ##########################
# ########## DATA ##########
# ##########################

def compute_angularities(events, beta = 2.0):

    y = []
    for x in events:
        zs = x[:,0]
        thetas = np.sqrt(np.sum(x[:,1:3] **2, axis = 1))
        y.append(np.sum(zs * np.power(thetas, beta)))

    return np.array(y)

if dataset == "qg":
    features = []
    X, Y = qg_jets.load(train+val+test, cache_dir="/n/holyscratch01/iaifi_lab/rikab/.energyflow")
    X = X[:,:,:3].astype(np.float32)
    for x in X:
        mask = x[:,0] > 0
        yphi_avg = average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()


    (z_train, z_val, z_test,
    p_train, p_val, p_test,
    Y_train, Y_val, Y_test) = data_split(X[:,:,0], X[:,:,1:], Y, val=val, test=test)

    X_train = [z_train, p_train]
    X_val = [z_val, p_val]
    X_test = [z_test, p_test]

elif dataset == "top":
    raise ValueError("Only qg supported")
    X_train, Y_train = toploader.load(cache_dir=topdir, dataset="train", num_data = train)
    X_test, Y_test = toploader.load(cache_dir=topdir, dataset="test", num_data=test)
    X_val, Y_val = toploader.load(cache_dir=topdir, dataset="val", num_data=test)

    def format(X):

        for x in X:
            mask = x[:,0] > 0
            yphi_avg = average(x[mask,1:3], weights=x[mask,0], axis=0)
            x[mask,1:3] -= yphi_avg
            x[mask,0] /= x[:,0].sum()

        return [X[:,:,0], X[:,:,1:3]]
    
    X_train = format(X_train)
    X_test = format(X_test)
    X_val = format(X_val)

    Y_train = to_categorical(Y_train, num_classes=2)
    Y_test = to_categorical(Y_test, num_classes=2)
    Y_val = to_categorical(Y_val, num_classes=2)


else:
    raise ValueError("dataset must be either `top` or `qg`!")


print("Data loaded!")





order_configs = {}
for p, order in enumerate(order_list):


    configs = []
    for i in range(num_samples):
        configs.append([Ls[0], F_width])

    order_configs['Order %d' % order] = np.asarray(configs).astype(np.int32)

   


## training


order_performance = dict()
order_histories = dict()

count = 0
for order in order_list:
    order_performance['Order '+str(order)] = []
    order_histories['Order '+str(order)] = []


output_array = []


for order in order_list:

    for (i, beta) in enumerate(betas):

        losses = []
        for n in range(num_models_to_train):

            # try:

                # info = order_configs['Order '+str(order)][i]
                # L, F = info
                L = 2
                Phi = 100
                F = 100
                print(L, F, order)

                model_name = f"O{order}_L{L}_2Phi{Phi}_3F{F}_{n}.keras"
                model = EFN_moment(**{'Phi_mapping_dim' : [input_dim,2],
                                        'output_dim' : 1, 'output_act' : 'linear',
                                        'Phi_sizes' : [], 'Phi_acts' : 'linear', "Phi_l1_regs" :  0,
                                        'F_sizes' : [], 'F_acts': 'linear', "F_l1_regs" :  0,
                                        'order' : k_order , 'architecture_type':'moment',
                                        'loss': 'mse','metrics': 'mse'}, summary = False, bias = False)

                angularities = compute_angularities(X, beta)
                (z_train, z_val, z_test,
                    p_train, p_val, p_test,
                    Y_train, Y_val, Y_test) = data_split(X[:,:,0], X[:,:,1:], angularities, val=val, test=test)

                X_train = [z_train, p_train]
                X_val = [z_val, p_val]
                X_test = [z_test, p_test]

                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8)
                history = model.fit(X_train, Y_train,
                                    epochs = epochs, batch_size = batch_size,
                                    validation_data = (X_val, Y_val),
                                    callbacks=[callback], verbose=0,)
                
                print(k_order, beta, model.evaluate(X_test, Y_test, verbose = 0)[1])
                model.save_weights(os.path.join(model_dir, model_name))

                losses.append(model.evaluate(X_test, Y_test, verbose = 0)[1])
                order_histories['Order '+str(order)].append(history.history)
        output_array.append(losses)


np.save(f"Data/angularities/O{k_order}_losses.npy", np.array(output_array))



np.save(os.path.join(run_dir , 'configs'), order_configs)
np.save(os.path.join(run_dir , 'histories') , order_histories)



