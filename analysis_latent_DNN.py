from ModelsContainer import ModelsContainer
from energyflow.archs.dnn import DNN
from numpy import average
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, to_categorical
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
val = 50000
test = 50000
k_order = int(sys.argv[1]) 
run_name = sys.argv[3]
dataset = sys.argv[2] # either "qg" or "top"


epochs = 50
batch_size=512
callbacks =None
verbose = 2


num_models_to_train = 3 ##number of models to use to make error bars
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


Ls = []
Ls = [1, 2, 3, 4, 5, 6, 7, 8, 10, 16, 32, 64, 128]

num_samples = len(Ls)


# max_L_per_order = [2**(8-k_order),]
# F_min, F_max = 100
# Phi_min, Phi_max = 100
# logN_max = 6.5


# ##########################
# ########## DATA ##########
# ##########################

def log_features(x):


    a = 0
    b = 1.0
    c = 0.005

    zs = x[:,0]

    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg

    rs = np.sqrt(np.sum(np.square(x[:,1:3]), axis = 1))
    ls = a + b*np.log(rs + c)

    # ls = rs

    l_list = []

    for n in range(128 + 1):

        l_list.append(np.sum(zs * np.nan_to_num(np.power(ls, n))))

    return np.array(l_list)

if dataset == "qg":
    features = []
    X, Y = qg_jets.load(train+val+test, cache_dir="/n/holyscratch01/iaifi_lab/rikab/.energyflow")
    X = X[:,:,:3].astype(np.float32)
    for x in X:
        mask = x[:,0] > 0
        yphi_avg = average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
        features.append(log_features(x))

    X = np.nan_to_num(np.array(features))

    (X_train, X_val, X_test,
    Y_train, Y_val, Y_test) = data_split(X, Y, val=val, test=test)


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
        configs.append([Ls[i], F_width])

    order_configs['Order %d' % order] = np.asarray(configs).astype(np.int32)

   


## training


order_performance = dict()
order_histories = dict()

count = 0
for order in order_list:
    order_performance['Order '+str(order)] = []
    order_histories['Order '+str(order)] = []

for order in order_list:
    for (i, L) in enumerate(Ls):

        # try:

            info = order_configs['Order '+str(order)][i]
            L, F = info
            print(L, F, order)

            (X_train, X_val, X_test, Y_train, Y_val, Y_test) = data_split(X[:,:(L + 1)], Y, val=val, test=test)

            scores = []
            for sample in range(num_models_to_train):

                model_name = f"DNN_L{L}_3F{F}"
                model = DNN(input_dim=L+1, dense_sizes=[F,F,F], metrics = [tf.keras.metrics.AUC()], acts='LeakyReLU', output_dim = 1, output_act = "sigmoid")
                print()
                print(i, order, info, model.count_params())
                print()


                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8)
                history = model.fit(X_train, Y_train,
                                            epochs = epochs, batch_size = batch_size,
                                            validation_data = (X_val, Y_val),
                                            callbacks=[callback], verbose=verbose,)
                model.save_weights(os.path.join(model_dir , model_name+ f"_{sample}.keras"))
                scores.append(roc_auc_score(Y_test, model.predict(X_test)))
            

            # container.test_meanstd(X_test = X_test, Y_test = Y_test, metric_function = roc_auc_score)
            num_params = model.count_params()
            
            order_performance['Order '+str(order)].append([num_params,np.mean(scores),np.std(scores)])
            order_histories['Order '+str(order)].append(history.history)
            count += 1

        # except:
        #     print(f"Effective latent dim too big!")
    
    order_performance['Order '+str(order)] = np.array(order_performance['Order '+str(order)])




file_to_save = {'configs' : order_configs, 'performance' : order_performance}
np.save(os.path.join(run_dir , 'datapoints'), file_to_save)
np.save(os.path.join(run_dir , 'configs'), order_configs)
np.save(os.path.join(run_dir , 'performance') , order_performance)
np.save(os.path.join(run_dir , 'histories') , order_histories)



