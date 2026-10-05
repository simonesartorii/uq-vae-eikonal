import numpy as np
import tensorflow as tf 
import os

def load_encoder_data(train_path, val_path, test_path, scale_path, n_sensors,
                      region_bounds=(0.5, 0.5), sensor_idx_path=None,
                      sensor_idx=None, seed=42, 
                      noise_level=0.0, dtype=tf.float32):
    """
    Loads FEM data for encoder training (ready for augmentation)
    Args:
        region_bounds: sensors are sampled in the region x >= x_min, y >= y_min
    """
    np.random.seed(42)
    tf.random.set_seed(42)

    U_SCALE = float(np.load(scale_path)[0])
    
    def _load_data(path):
        data = np.load(path)
        coords = data['coords']
        values = data['values_clean'] if 'values_clean' in data else data['values']
        noise = data['noise'] if 'noise' in data else None
        
        cx_0, cy_0 = coords[0, 2], coords[0, 3]
        pts_per_sim = 1
        for i in range(1, len(coords)):
            if coords[i, 2] != cx_0 or coords[i, 3] != cy_0:
                pts_per_sim = i
                break
        if pts_per_sim == 1 and len(coords) > 1 and coords[1,2] == cx_0:
            pts_per_sim = len(coords)
            
        n_sim = len(values) // pts_per_sim
        Y_full = values.reshape(n_sim, pts_per_sim) / U_SCALE
        N_full = noise.reshape(n_sim, pts_per_sim) / U_SCALE if noise is not None else None
        U_centers = coords[::pts_per_sim, 2:4] 
        return coords, Y_full, U_centers, N_full, pts_per_sim

    coords_train, Y_train_full, u_train, _, pts_per_sim = _load_data(train_path)
    _, Y_val_full, u_val, N_val_full, _ = _load_data(val_path)
    _, Y_test_full, u_test, N_test_full, _ = _load_data(test_path)

    if sensor_idx_path is not None and os.path.exists(sensor_idx_path):
        sensor_idx = np.load(sensor_idx_path)
        print(f"[SENSORS] Used sensors saved in: {sensor_idx_path}")
    elif pts_per_sim == n_sensors:
        sensor_idx = np.arange(n_sensors)
    else:
        x_min, y_min = region_bounds
        region_mask = (coords_train[:pts_per_sim, 0] >= x_min) & (coords_train[:pts_per_sim, 1] >= y_min)
        valid_indices = np.where(region_mask)[0]
        sensor_idx = np.random.choice(valid_indices, n_sensors, replace=False)
        if sensor_idx_path is not None:
            np.save(sensor_idx_path, sensor_idx)
            print(f"[SENSORS] New sensors saved in: {sensor_idx_path}")

    x_obs = tf.convert_to_tensor(coords_train[sensor_idx, 0:2], dtype=dtype)

    # clean training set
    y_train_clean = tf.convert_to_tensor(Y_train_full[:, sensor_idx], dtype=dtype)
    u_train = tf.convert_to_tensor(u_train, dtype=dtype)
    
    # noisy val and test set 
    y_val = tf.convert_to_tensor(Y_val_full[:, sensor_idx], dtype=dtype)
    u_val = tf.convert_to_tensor(u_val, dtype=dtype)
    y_test = tf.convert_to_tensor(Y_test_full[:, sensor_idx], dtype=dtype)
    u_test = tf.convert_to_tensor(u_test, dtype=dtype)

    if N_val_full is not None:
        y_val += tf.convert_to_tensor(N_val_full[:, sensor_idx], dtype=dtype)
        y_test += tf.convert_to_tensor(N_test_full[:, sensor_idx], dtype=dtype)
    elif noise_level > 0.0:
        y_val += tf.random.normal(shape=tf.shape(y_val), mean=0.0, stddev=noise_level, dtype=dtype)
        y_test += tf.random.normal(shape=tf.shape(y_test), mean=0.0, stddev=noise_level, dtype=dtype)

    print(f"Shape y_train: {y_train_clean.shape}  |  Shape u_train: {u_train.shape}")
    print(f"Shape y_val:   {y_val.shape}  |  Shape u_val:   {u_val.shape}")
    print(f"Shape y_test:  {y_test.shape}  |  Shape u_test:  {u_test.shape}")

    return (y_train_clean, u_train), (y_val, u_val), (y_test, u_test), x_obs, sensor_idx, U_SCALE


def load_vae_inference_data(data_path, scale_path, sensor_idx_path, noise_level=0.0, dtype=tf.float64):
    """
    Loads FEM data for VAE inference, handling noise injection.
    """

    U_SCALE = float(np.load(scale_path)[0])
    print(f"U_SCALE (loaded): {U_SCALE}")
 
    data = np.load(data_path)
    coords = data['coords']
    values = data['values_clean'] if 'values_clean' in data else data['values']
    noise = data['noise'] if 'noise' in data else None

    cx_0, cy_0 = coords[0, 2], coords[0, 3]
    pts_per_sim = 1
    for i in range(1, len(coords)):
        if coords[i, 2] != cx_0 or coords[i, 3] != cy_0:
            pts_per_sim = i
            break
            
    if pts_per_sim == 1 and len(coords) > 1 and coords[1, 2] == cx_0:
        pts_per_sim = len(coords)
            
    n_sim = len(values) // pts_per_sim
    Y_full = values.reshape(n_sim, pts_per_sim) / U_SCALE
    N_full = noise.reshape(n_sim, pts_per_sim) / U_SCALE if noise is not None else None
    U = coords[::pts_per_sim, 2:4]  
 
    sensor_idx = np.load(sensor_idx_path)
    x_obs = tf.convert_to_tensor(coords[:pts_per_sim, 0:2][sensor_idx, :], dtype=dtype)
 
    y_test = tf.convert_to_tensor(Y_full[:, sensor_idx], dtype=dtype)
    
    if N_full is not None:
        n_test = tf.convert_to_tensor(N_full[:, sensor_idx], dtype=dtype)
        y_test += n_test
        print("Applied pre-computed consistent noise for inference.")
    elif noise_level > 0.0:
        y_test += tf.random.normal(shape=tf.shape(y_test), mean=0.0, stddev=noise_level, dtype=dtype)
        print(f"Injected Gaussian noise with stddev: {noise_level} for inference.")
        
    u_test = tf.convert_to_tensor(U, dtype=dtype)
 
    print(f"Shape y_test: {y_test.shape}  |  Shape u_test: {u_test.shape}")
 
    return y_test, u_test, x_obs, sensor_idx, U_SCALE