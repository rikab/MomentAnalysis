from energyflow.archs.moment import EFN_moment, PFN_moment
from sklearn.metrics import roc_auc_score
from utils.data_utils import load_data
try:
    from config import base_dir
except:
    raise ValueError("Please specify a base analysis directory, base_dir, in config.py!")

import numpy as np
import tensorflow as tf
import sys
import os


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

# Turn off TEST_MODE to run on full dataset
TEST_MODE = True

# Parameters 
train = 1000000
val = 50000
test = 50000
k_order = int(sys.argv[1]) 
dataset = sys.argv[2] # either "qg" or "top"
run_name = sys.argv[3]
architecture = sys.argv[4] # either EFN or PFN
categorical = False # True to use one-hot encoding for labels


num_models_to_train = 3 #number of models to use to make error bars
output_dim = 1

max_L = 2**(8-k_order) 
F_width = 100
Phi_width = 100

# check if architecture is valid
if architecture not in ["EFN", "PFN"]:  
    raise ValueError("Architecture must be either `EFN` or `PFN`!")
is_EFN = architecture == "EFN"
input_dim = 2 if is_EFN else 3


epochs = 50
batch_size=512
callbacks =None
verbose = 2


# Directory Handling
run_dir = os.path.join(base_dir, "Data", run_name)
run_dir = os.path.join(run_dir, f"order_{k_order}")
model_dir = os.path.join(run_dir, "Models")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
    os.makedirs(model_dir, exist_ok=True)
print("Run directory: ", run_dir)

if TEST_MODE:
    epochs = 50
    num_models_to_train = 1
    train = 10000
    val = 1000
    test = 1000
    F_width = 25
    Phi_width = 25
    max_L = 1

###########
Ls = []
j = max_L
while j >= 1:
    Ls.append(j)
    j = j / 2

num_samples = len(Ls)



# ##########################
# ########## DATA ##########
# ##########################

(X_train, X_val, X_test,), (Y_train, Y_val, Y_test,) = load_data(dataset, train, val, test, 
                                                                 EFN_format=is_EFN, 
                                                                 categorical=categorical)
print("Data loaded!")


# ##########################
# ########## MODEL #########
# ##########################

# Choose model architecture
Model = EFN_moment if is_EFN else PFN_moment

# Initialize model configs to train
configs = []
for i in range(num_samples):

    L = Ls[i]
    F = F_width
    Phi = Phi_width

    output_dim = 1 if not categorical else 2
    output_act = 'sigmoid' if not categorical else 'softmax'
    loss = 'binary_crossentropy' if not categorical else 'categorical_crossentropy'


    config = {'Phi_mapping_dim' : [input_dim,L],
                                    'output_dim' : output_dim, 'output_act' : output_act,
                                    'Phi_sizes' : [Phi, Phi], 'Phi_acts' : 'LeakyReLU', "Phi_l1_regs" :  1e-6,
                                    'F_sizes' : [F,F,F], 'F_acts': 'LeakyReLU', "F_l1_regs" :  1e-6,
                                    'order' : k_order , 'architecture_type':'moment',
                                    'loss': loss,
                                    # 'save_weights_only' : True,
                                    }
    configs.append(config)


# ##########################
# ########## TRAIN #########
# ##########################

aucs = []
histories = []

for config in configs:

    aucs_i = []
    histories_i = []

    for j in range(num_models_to_train):

        # Filepath to save model
        model_name = f"O{k_order}_L{L}_2Phi{Phi}_3F{F}_{j}.keras"
        model_filepath = os.path.join(model_dir , model_name)

        # Initialize model
        model = Model(**config, filepath = model_filepath, metrics =  ['acc', tf.keras.metrics.AUC()], summary=False)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8)


        # Train model
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data = (X_val, Y_val), verbose=verbose, callbacks=[callback,])
    
        # Test model
        preds = model.predict(X_test)
        auc = roc_auc_score(Y_test, preds)
        aucs_i.append(auc)
        histories_i.append(history.history)

    # Save results
    aucs_i = np.array(aucs_i)
    aucs.append(aucs_i)
    histories.append(histories_i)

# Save files
np.save(os.path.join(run_dir , 'configs.npy'), configs)
np.save(os.path.join(run_dir , 'performance.npy') , aucs)
np.save(os.path.join(run_dir , 'histories.npy') , histories)


print("Done!")