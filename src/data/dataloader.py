import numpy as np
import tensorflow as tf
import os

def _base_load_fem_data(data_path, val_path=None, test_path=None,
                        scale_path='data/models/u_scale_var.npy', 
                        compute_scale=True,
                        n_train=None, n_val=None, n_test=None,
                        train_split=0.8, val_split=0.1,
                        noise_level=0.0, dtype=tf.float32):
    """
    Base function for loading fem data, scaling, split, noise injeciton.
    Args:
        compute_scale: boolean for computing/loading U_SCALE
        n_train: number of training samples
        n_val: number of validation samples
        n_test: number of test samples
        train_split: % of training samples
        val_split: % of validation samples
    """
    
    if val_path is None and test_path is None:
        data = np.load(data_path)
        coords, values = data['coords'], data['values']
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
        
        x_val = coords[all_idx[n_train:n_train+n_val]]
        u_val = values[all_idx[n_train:n_train+n_val]]
        
        x_test = coords[all_idx[n_train+n_val:n_train+n_val+n_test]]
        u_test = values[all_idx[n_train+n_val:n_train+n_val+n_test]]

    else:
        data_train = np.load(data_path)
        x_train, u_train = data_train['coords'], data_train['values']

        data_val = np.load(val_path)
        x_val, u_val = data_val['coords'], data_val['values']

        data_test = np.load(test_path)
        x_test, u_test = data_test['coords'], data_test['values']

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

    # noise injection 
    if noise_level > 0.0:
        noise = tf.random.normal(shape=tf.shape(u_train), mean=0.0, stddev=noise_level, dtype=dtype)
        u_train = u_train + noise
        print(f"Injected Gaussian noise with stddev: {noise_level}")

    print(f"Shape x_train: {x_train.shape}")
    print(f"Shape x_val:   {x_val.shape}")
    print(f"Shape x_test:  {x_test.shape}")

    return (x_train, u_train), (x_val, u_val), (x_test, u_test), U_SCALE

def load_fem_data(data_path, scale_save_path, number_type=tf.float32,
                  num_train=2000, num_val=500, num_test=500, noise_level=0.0):

    print(f"Loading FEM data from:\n{data_path}")
    return _base_load_fem_data(data_path, val_path=None, test_path=None, 
                               scale_path=scale_save_path, compute_scale=True,
                               n_train=num_train, n_val=num_val, n_test=num_test,
                               noise_level=noise_level, dtype=number_type)
    

def load_parametric_fem_data(train_path, val_path, test_path, scale_save_path, 
                             noise_level=0.0, dtype=tf.float32):

    print(f"Loading Parametric FEM data from:\n{train_path}\n{val_path}\n{test_path}")
    return _base_load_fem_data(data_path=train_path, val_path=val_path, test_path=test_path, 
                               scale_path=scale_save_path, compute_scale=True,
                               noise_level=noise_level, dtype=dtype)

def load_parametric_fem_data_lbfgs(train_path, val_path, test_path, scale_load_path, 
                                   noise_level=0.0, n_lbfgs_data=15000, dtype=tf.float64):
   
    print(f"Loading Parametric FEM data for L-BFGS from:\n{train_path}\n{val_path}\n{test_path}")
    return _base_load_fem_data(data_path=train_path, val_path=val_path, test_path=test_path, 
                               scale_path=scale_load_path, compute_scale=False, 
                               n_train=n_lbfgs_data, noise_level=noise_level, dtype=dtype)

def load_encoder_data(train_path, val_path, test_path, scale_path, n_grid, n_sensors,
                      region_bounds=(0.5, 0.5), sensor_idx_path=None,
                      noise_level=0.0, dtype=tf.float32):
    """
    Loads FEM data for encoder training. 
    Carica i dati FEM su griglia già splittati per l'addestramento dell'encoder.
    Estrae le osservazioni (Y) sui sensori selezionati e i centri (U).
    """
    # loading scaling factor 
    U_SCALE = float(np.load(scale_path)[0])
    print(f"U_SCALE (loaded): {U_SCALE}")
    D_y_full = n_grid * n_grid

    # helper to extract grid and centers 
    def _load_grid(path):
        data = np.load(path)
        coords, values = data['coords'], data['values']
        n_sim = len(values) // D_y_full
        Y_full = values.reshape(n_sim, D_y_full) / U_SCALE
        U_centers = coords[::D_y_full, 2:4] 
        return coords, Y_full, U_centers

    # loading splits 
    coords_train, Y_train_full, u_train = _load_grid(train_path)
    _, Y_val_full, u_val = _load_grid(val_path)
    _, Y_test_full, u_test = _load_grid(test_path)

    # sensor selection 
    x_min, y_min = region_bounds
    region_mask = (coords_train[:D_y_full, 0] >= x_min) & (coords_train[:D_y_full, 1] >= y_min)
    valid_indices = np.where(region_mask)[0]
    sensor_idx = np.random.choice(valid_indices, n_sensors, replace=False)
    if sensor_idx_path is not None:
        np.save(sensor_idx_path, sensor_idx)

    x_obs = tf.convert_to_tensor(coords_train[sensor_idx, 0:2], dtype=dtype)

    # 4. Creazione tensori Y (solo sensori) e U (centri)
    y_train = tf.convert_to_tensor(Y_train_full[:, sensor_idx], dtype=dtype)
    u_train = tf.convert_to_tensor(u_train, dtype=dtype)
    
    y_val = tf.convert_to_tensor(Y_val_full[:, sensor_idx], dtype=dtype)
    u_val = tf.convert_to_tensor(u_val, dtype=dtype)

    y_test = tf.convert_to_tensor(Y_test_full[:, sensor_idx], dtype=dtype)
    u_test = tf.convert_to_tensor(u_test, dtype=dtype)

    # 5. Iniezione rumore solo su training
    if noise_level > 0.0:
        noise = tf.random.normal(shape=tf.shape(y_train), mean=0.0, stddev=noise_level, dtype=dtype)
        y_train = y_train + noise
        print(f"Injected Gaussian noise with stddev: {noise_level}")

    print(f"Shape y_train: {y_train.shape}  |  Shape u_train: {u_train.shape}")
    print(f"Shape y_val:   {y_val.shape}  |  Shape u_val:   {u_val.shape}")
    print(f"Shape y_test:  {y_test.shape}  |  Shape u_test:  {u_test.shape}")

    return (y_train, u_train), (y_val, u_val), (y_test, u_test), x_obs, sensor_idx, U_SCALE


def load_vae_inference_data(data_path, scale_path, sensor_idx_path, n_grid, dtype=tf.float64):
    """
    Loads FEM grid data vae inference.
 
    Args:
        data_path: path of FEM grid 
        scale_path: path U_SCALE
        sensor_idx_path: path of sensor indices previously saved
        n_grid: number of points per side of the FEM grid (D_y_full = n_grid ^ 2)
    """
    U_SCALE = float(np.load(scale_path)[0])
    print(f"U_SCALE (loaded): {U_SCALE}")
    D_y_full = n_grid * n_grid
 
    data = np.load(data_path)
    coords, values = data['coords'], data['values']
    n_sim = len(values) // D_y_full
    Y_full = values.reshape(n_sim, D_y_full) / U_SCALE
    U = coords[::D_y_full, 2:4]  # [cx, cy]
 
    sensor_idx = np.load(sensor_idx_path)
    x_obs = tf.convert_to_tensor(coords[:D_y_full, 0:2][sensor_idx, :], dtype=dtype)
 
    y_test = tf.convert_to_tensor(Y_full[:, sensor_idx], dtype=dtype)
    u_test = tf.convert_to_tensor(U, dtype=dtype)
 
    print(f"Shape y_test: {y_test.shape}  |  Shape u_test: {u_test.shape}")
 
    return y_test, u_test, x_obs, sensor_idx, U_SCALE