from ModelsContainer import ModelsContainer
from numpy import average
from energyflow.datasets import qg_jets
from energyflow.utils import data_split
from sklearn.metrics import roc_auc_score
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
val = 5000
test = 5000
k_order = int(sys.argv[1]) 
run_name = sys.argv[2]
dataset = sys.argv[2] # either "qg" or "top"


epochs = 50
batch_size=2048
callbacks =None
verbose = 2


# Directory Handling
base_dir = "/n/home01/rikab/MomentAnalysis/Data"
run_dir = os.path.join(base_dir, run_name)
model_dir = os.path.join(run_dir, "Models")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
    os.makedirs(model_dir, exist_ok=True)






# Parameters 
train = 1000000
val = 5000
test = 5000
k_order = int(sys.argv[1]) 

epochs = 50
batch_size=2048
callbacks =None
verbose = 2

###########

num_models_to_train = 3 ##number of models to use to make error bars
order_list = [k_order,] #
input_dim = 2
output_dim = 1


max_L = 2**(8-k_order)
F_width = 100
Phi_width = 100


Ls = []
j = max_L
while j >= 1:
    Ls.append(j)
    j = j / 2

num_samples = len(Ls)


# max_L_per_order = [2**(8-k_order),]
# F_min, F_max = 100
# Phi_min, Phi_max = 100
# logN_max = 6.5


# ##########################
# ########## DATA ##########
# ##########################

if dataset == "qg":
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
    X_train, Y_train = toploader.load(cache_dir=dir, dataset="train", num_data = train)
    X_test, Y_test = toploader.load(cache_dir=dir, dataset="test", num_data=test)
    X_val, Y_val = toploader.load(cache_dir=dir, dataset="val", num_data=test)

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

else:
    raise ValueError("dataset must be either `top` or `qg`!")

print("Data loaded!")





order_configs = {}
for p, order in enumerate(order_list):


    configs = []
    for i in range(num_samples):
        configs.append([Ls[i], F_width, Phi_width])

    order_configs['Order %d' % order] = np.asarray(configs).astype(np.int32)

   


## training


order_performance = dict()
order_histories = dict()

count = 0
for order in order_list:
    order_performance['Order '+str(order)] = []
    order_histories['Order '+str(order)] = []

for order in order_list:
    for i in range(num_samples):
        info = order_configs['Order '+str(order)][i]
        L, F, Phi = info
        print(L, F, Phi, order)

        model_name = f"O{order}_L{L}_2Phi{Phi}_3F{F}"
        container = ModelsContainer(**{'Phi_mapping_dim' : [input_dim,L],
                                      'output_dim' : 1, 'output_act' : 'sigmoid',
                                      'Phi_sizes' : [Phi, Phi], 'Phi_acts' : 'LeakyReLU', "Phi_l1_regs" :  1e-7,
                                      'F_sizes' : [F,F,F], 'F_acts': 'LeakyReLU', "F_l1_regs" :  1e-7,
                                      'order' : order , 'architecture_type':'moment',
                                      'loss': 'binary_crossentropy','metrics': 'acc','metrics': ['acc', tf.keras.metrics.AUC()]})
        print()
        print(i, order, info, container.num_params)
        print()
        mean, std = container.train_models(num_models = num_models_to_train,
                                    X_train = X_train, Y_train = Y_train,
                                    epochs = epochs, batch_size = batch_size,
                                    path = os.path.join(model_dir , model_name),
                                    validation_data = (X_val, Y_val),
                                    callbacks=None, verbose=verbose,
                                    metric_function = roc_auc_score)
        # container.test_meanstd(X_test = X_test, Y_test = Y_test, metric_function = roc_auc_score)
        num_params = container.num_params
        
        order_performance['Order '+str(order)].append([num_params,mean,std])
        order_histories['Order '+str(order)].append(container.histories)

        # Just for fun, update the plot
        # plot(order_performance, '/n/home01/rikab/moment/Plots/gpu_%d.pdf' % count)
        container.save_model_weights(os.path.join(model_dir , model_name))
        count += 1
   
    order_performance['Order '+str(order)] = np.array(order_performance['Order '+str(order)])




file_to_save = {'configs' : order_configs, 'performance' : order_performance}
np.save(run_dir + 'datapoints', file_to_save)
np.save(run_dir + 'configs', order_configs)
np.save(run_dir + 'performance' , order_performance)
np.save(run_dir + 'histories' , order_histories)

