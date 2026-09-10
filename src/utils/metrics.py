import numpy as np
import tensorflow as tf

def compute_and_print_metrics(u_fem, u_pinn, U_SCALE):
    """
    Calcola e stampa le metriche di errore sul test set:
    - MSE adimensionale
    - MAE fisico (in millisecondi)
    """
    # 1. Calcolo MSE Adimensionale
    u_pinn_adim = u_pinn / float(U_SCALE)
    u_fem_adim = u_fem / float(U_SCALE)
    test_mse = np.mean(np.square(u_pinn_adim - u_fem_adim))

    # 2. Calcolo MAE Fisico in millisecondi
    test_mae = np.mean(np.abs(u_pinn - u_fem)) * 1000.0

    # 3. Stampe a schermo
    print(f"Test MSE (Adimensionale) : {test_mse:.6e}")
    print(f"Test MAE (Fisico)        : {test_mae:.4f} ms")
    
    return test_mse, test_mae


def compute_test_metrics(model, coords_test, values_test, U_SCALE, dtype=tf.float32):
    """
    Computes the Relative L2 Error and Mean Absolute Error (MAE) 
    between the PINN predictions and the exact FEM ground truth.
    
    Returns:
        tuple: (l2_error, mae_error)
    """
    # 1. PINN predictions
    coords_tensor = tf.convert_to_tensor(coords_test, dtype=dtype)
    u_pred_test = model(coords_tensor).numpy().flatten() * U_SCALE
    
    # Assicuriamoci che anche i valori reali siano 1D per evitare bug di shape (N,1) vs (N,)
    values_test = values_test.flatten()

    # 2. Compute Errors
    # Relative L2 error: ||U_pred - U_true||_2 / ||U_true||_2
    l2_error = np.linalg.norm(u_pred_test - values_test) / np.linalg.norm(values_test)
    
    # Mean Absolute Error (Physical)
    mae_error = np.mean(np.abs(u_pred_test - values_test)) * 1000

    # 3. Print Results
    print(f"l2 Relative Error vs FEM Ground Truth: {l2_error:.4%}")
    print(f"Mean Absolute Error (Physical): {mae_error:.4f} ms")
    
    return float(l2_error), float(mae_error)