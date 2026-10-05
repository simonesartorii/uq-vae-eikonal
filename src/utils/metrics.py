import numpy as np
import tensorflow as tf

def predict_in_batches(model, coords, batch_size=500000, dtype=tf.float32):
    """Evaluates the model on coords (N, d) in chunks"""
    out = []
    for i in range(0, len(coords), batch_size):
        batch = tf.convert_to_tensor(coords[i:i + batch_size], dtype=dtype)
        out.append(model(batch).numpy())
    return np.concatenate(out, axis=0)

def compute_and_print_metrics(u_fem, u_pinn, U_SCALE):
    """
    Computes and prints error metrics on the test set:
    - Dimensionless MSE
    - Physical MAE (in milliseconds)
    """
    # Dimensionless MSE
    u_pinn_adim = u_pinn / float(U_SCALE)
    u_fem_adim = u_fem / float(U_SCALE)
    test_mse = np.mean(np.square(u_pinn_adim - u_fem_adim))

    # Physical MAE in milliseconds
    test_mae = np.mean(np.abs(u_pinn - u_fem)) * 1000.0

    print(f"Test MSE (Dimensionless) : {test_mse:.6e}")
    print(f"Test MAE (Physical)      : {test_mae:.4f} ms")
    
    return test_mse, test_mae

def compute_test_metrics(model, coords_test, values_test, U_SCALE, grads_test=None, dtype=tf.float32):
    """
    Computes the Relative L2 Error, Mean Absolute Error (MAE).
    
    Returns:
        tuple: (l2_error, mae_error, h1_error)
    """
    coords_tensor = tf.convert_to_tensor(coords_test, dtype=dtype)
    
    # PINN predictions and spatial gradients
    with tf.GradientTape() as tape:
        tape.watch(coords_tensor)
        u_pred_tensor = model(coords_tensor)
        
    u_pred_test = u_pred_tensor.numpy().flatten() * float(U_SCALE)
    values_test = values_test.flatten()

    # Compute L2 and MAE Errors
    l2_error = np.linalg.norm(u_pred_test - values_test) / np.linalg.norm(values_test)
    mae_error = np.mean(np.abs(u_pred_test - values_test)) * 1000.0 # to milliseconds

    print(f"L2 Relative Error vs FEM   : {l2_error:.4%}")
    print(f"Mean Absolute Error (Phys) : {mae_error:.4f} ms")

    return float(l2_error), float(mae_error)