# ############################
# ########## IMPORTS #########
# ############################

from energyflow.datasets import qg_jets
from energyflow.utils import data_split, to_categorical
import toploader
import numpy as np

try:
    from config import cache_dirs
except ImportError:
    print("Warning: cache_dirs not found in config.py!")



def load_data(dataset, train, val, test, EFN_format = True, categorical = False, cache_dir = None):
    """Load data from a dataset

    Args:
        dataset (_type_): _description_
        train (_type_): _description_
        val (_type_): _description_
        test (_type_): _description_
        EFN_format (bool, optional): _description_. Defaults to True.
        categorical (bool, optional): _description_. Defaults to False.
        cache_dir (_type_, optional): _description_. Defaults to None.

    Raises:
        KeyError: _description_
        ValueError: _description_

    Returns:
        
    """

    # Specify cache directory
    if cache_dir is None:
        try:
            cache_dir = cache_dirs[dataset]
        except KeyError:
            raise KeyError(f"Please specify a cache directory for the {dataset} dataset!")


    
    # Quark-Gluon dataset
    if dataset == "qg":
        

        X, Y = qg_jets.load(train+val+test, cache_dir=cache_dir)
        X = X[:,:,:3].astype(np.float32)

        # Normalize and center
        for x in X:
            mask = x[:,0] > 0
            yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
            x[mask,1:3] -= yphi_avg
            x[mask,0] /= x[:,0].sum()

        if categorical:
            Y = to_categorical(Y, num_classes=2)

        # Split the data depending on whether we want EFN format or not
        if EFN_format:
            (z_train, z_val, z_test,
            p_train, p_val, p_test,
            Y_train, Y_val, Y_test) = data_split(X[:,:,0], X[:,:,1:], Y, val=val, test=test)

            X_train = [z_train, p_train]
            X_val = [z_val, p_val]
            X_test = [z_test, p_test]
        else:
            (X_train, X_val, X_test,
            Y_train, Y_val, Y_test) = data_split(X, Y, val=val, test=test)

        # Return
        return (X_train, X_val, X_test,), (Y_train, Y_val, Y_test,)

    # Top dataset
    elif dataset == "top":
        X_train, Y_train = toploader.load(cache_dir=cache_dir, dataset="train", num_data = train)
        X_test, Y_test = toploader.load(cache_dir=cache_dir, dataset="test", num_data=test)
        X_val, Y_val = toploader.load(cache_dir=cache_dir, dataset="val", num_data=test)

        # Normalize and center
        def format(X):
            for x in X:
                mask = x[:,0] > 0
                yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
                x[mask,1:3] -= yphi_avg
                x[mask,0] /= x[:,0].sum()

            # Reformat if EFN_format specified
            if EFN_format:
                X = [X[:,:,0], X[:,:,1:3]] 
            return X
        
        X_train = format(X_train)
        X_test = format(X_test)
        X_val = format(X_val)

        if categorical:
            Y_train = to_categorical(Y_train, num_classes=2)
            Y_test = to_categorical(Y_test, num_classes=2)
            Y_val = to_categorical(Y_val, num_classes=2)

        # Return
        return (X_train, X_val, X_test,), (Y_train, Y_val, Y_test,)


    else:
        raise ValueError("dataset must be either `top` or `qg`!")