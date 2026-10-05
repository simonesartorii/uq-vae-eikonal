import numpy as np
import tensorflow as tf

def _base_load_fem_data(data_path, val_path=None, test_path=None,
                        scale_path=None, 
                        compute_scale=True,
                        n_train=None, n_val=None, n_test=None,
                        train_split=0.8, val_split=0.1,
                        dtype=tf.float32):
    """
    Base function for loading fem data, scaling, split, noise injeciton.
    Args:
        compute_scale: if True, compute the scaling factor from the training data and save it to scale_path. If False, load the scaling factor from scale_path.
    """
    
    if val_path is None and test_path is None:
        data = np.load(data_path)
        coords = data['coords']
        values = data['values_clean'] if 'values_clean' in data else data['values'] # retrocompatibility
        noise_data = data['noise'] if 'noise' in data else None
        N_tot = len(values)

        all_idx = np.arange(N_tot)
        np.random.shuffle(all_idx)

        # idx split
        if n_train is None and n_val is None and n_test is None:
            n_train = int(train_split * N_tot)
            n_val = int(val_split * N_tot)
            n_test = N_tot - n_train - n_val

        x_train = coords[all_idx[:n_train]]
        u_train = values[all_idx[:n_train]]
        noise_train = noise_data[all_idx[:n_train]] if noise_data is not None else None
        
        x_val = coords[all_idx[n_train:n_train+n_val]]
        u_val = values[all_idx[n_train:n_train+n_val]]
        noise_val = noise_data[all_idx[n_train:n_train+n_val]] if noise_data is not None else None
        
        x_test = coords[all_idx[n_train+n_val:n_train+n_val+n_test]]
        u_test = values[all_idx[n_train+n_val:n_train+n_val+n_test]]
        noise_test = noise_data[all_idx[n_train+n_val:n_train+n_val+n_test]] if noise_data is not None else None

    else:
        data_train = np.load(data_path)
        x_train = data_train['coords']
        u_train = data_train['values_clean'] if 'values_clean' in data_train else data_train['values']
        noise_train = data_train['noise'] if 'noise' in data_train else None

        data_val = np.load(val_path)
        x_val = data_val['coords']
        u_val = data_val['values_clean'] if 'values_clean' in data_val else data_val['values']
        noise_val = data_val['noise'] if 'noise' in data_val else None

        data_test = np.load(test_path)
        x_test = data_test['coords']
        u_test = data_test['values_clean'] if 'values_clean' in data_test else data_test['values']
        noise_test = data_test['noise'] if 'noise' in data_test else None

        # subsampling and shuffling 
        N_train = len(u_train)
        if n_train is not None:
            actual_n = min(n_train, N_train)
            idx_train = np.random.choice(N_train, actual_n, replace=False)
        else:
            idx_train = np.arange(N_train)
            np.random.shuffle(idx_train)

        x_train = x_train[idx_train]
        u_train = u_train[idx_train]
        if noise_train is not None:
            noise_train = noise_train[idx_train]

    # noise injection 
    if noise_train is not None:
        u_train = u_train + noise_train
        u_val = u_val + noise_val
        u_test = u_test + noise_test
        print("Applied pre-computed noise.")
  
    # loading/computing scaling factor 
    if compute_scale:
        U_SCALE = np.max(u_train) 
        np.save(scale_path, np.array([U_SCALE]))
        print(f"U_SCALE: {U_SCALE}")
    else:
        U_SCALE = np.load(scale_path)[0]
        print(f"U_SCALE (loaded): {U_SCALE}")

    x_train = tf.convert_to_tensor(x_train, dtype=dtype)
    u_train = tf.convert_to_tensor(u_train / U_SCALE, dtype=dtype)

    x_val = tf.convert_to_tensor(x_val, dtype=dtype)
    u_val = tf.convert_to_tensor(u_val / U_SCALE, dtype=dtype)

    x_test = tf.convert_to_tensor(x_test, dtype=dtype)
    u_test = tf.convert_to_tensor(u_test / U_SCALE, dtype=dtype)


    print(f"Shape x_train: {x_train.shape}")
    print(f"Shape x_val:   {x_val.shape}")
    print(f"Shape x_test:  {x_test.shape}")

    return (x_train, u_train), (x_val, u_val), (x_test, u_test), U_SCALE


def load_fem_data(data_path, scale_save_path, number_type=tf.float32,
                  num_train=2000, num_val=500, num_test=500):
    """
    Function to load FEM data in non-parametric pipelines.
    """
    print(f"Loading FEM data from:\n{data_path}")
    return _base_load_fem_data(data_path, val_path=None, test_path=None, 
                               scale_path=scale_save_path, compute_scale=True,
                               n_train=num_train, n_val=num_val, n_test=num_test,
                               dtype=number_type)
    

def load_parametric_fem_data(train_path, val_path, test_path, scale_save_path, 
                             dtype=tf.float32):
    """
    Function to load FEM data in parametric pipelines (in the Adam optimization phase).
    """
    print(f"Loading Parametric FEM data from:\n{train_path}\n{val_path}\n{test_path}")
    return _base_load_fem_data(data_path=train_path, val_path=val_path, test_path=test_path, 
                               scale_path=scale_save_path, compute_scale=True,
                               dtype=dtype)

def load_parametric_fem_data_lbfgs(train_path, val_path, test_path, scale_load_path, 
                                   n_lbfgs_data=15000, dtype=tf.float64):
    """
    Function to load FEM data in parametric pipelines (in the L-BFGS optimization phase).
    """
    print(f"Loading Parametric FEM data for L-BFGS from:\n{train_path}\n{val_path}\n{test_path}")
    return _base_load_fem_data(data_path=train_path, val_path=val_path, test_path=test_path, 
                               scale_path=scale_load_path, compute_scale=False, 
                               n_train=n_lbfgs_data, dtype=dtype)
