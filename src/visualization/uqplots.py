import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from src.visualization.pdeplots import N_GRID

def plot_posterior_ellipse(u_pred_mean, u_pred_cov, u_true, sample_idx, n_std=2.0):
    """
    Plots the confidence ellipse of the posterior q(z|y) for a sample against the true scar center 
 
    Args:
        u_pred_mean: posterior means for all validation samples (N,2) 
        u_pred_cov: posterior covariances for all validation samples (N, 2, 2)
        u_true: true [cx, cy] for all validation samples (N, 2) 
        sample_idx: index of the sample to plot
        n_std: number of standard deviations for the confidence ellipse
    """
    mean = u_pred_mean[sample_idx]
    cov = u_pred_cov[sample_idx]
    true_val = u_true[sample_idx]
 
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
 
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
 
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
 
    ellip = Ellipse(xy=mean, width=width, height=height, angle=angle,
                     edgecolor='blue', fc='blue', alpha=0.2, lw=1.5)
    ax.add_patch(ellip)
 
    ax.scatter(*mean, color='blue', marker='o', s=60, label='Posterior Mean')
    ax.scatter(*true_val, color='red', marker='x', s=80, linewidth=2, label='True Scar Center')
 
    ax.set_title(f'Posterior Distribution for a single scar ({n_std}$\\sigma$)')
    ax.set_xlabel('cx')
    ax.set_ylabel('cy')
    ax.legend()
    plt.tight_layout()
    plt.show()


def propagate_and_plot_uncertainty(decoder_pinn, qz_val, u_pred_mean, u_true, sample_idx,
                                    U_SCALE, K_samples=100, N_fem=None, lato_cicatrice=0.2,
                                    number_type=tf.float64):
    """
    plots mean field prediction and relative predictive uncertainty for a validation sample 
    
    Args:
        decoder_pinn: pretrained decoder (frozen)
        qz_val: distribution returned by the encoder for the validation set 
        u_pred_mean: posterior means for all validation samples (N,2)
        u_true: true [cx, cy] for all validation samples (N,2)
        sample_idx: index of the sample to plot
        U_SCALE: scaling to revert normalization of decoder 
        K_samples: number of posterior samples 
        N_fem: number of points per side (N_grid from config)
        lato_cicatrice: scar side length
    """
    if N_fem is None:
        N_fem = N_GRID
 
    true_xy = u_true[sample_idx]
    true_cx, true_cy = true_xy.numpy() if hasattr(true_xy, 'numpy') else true_xy
    print(f"Propagazione incertezza per il campione {sample_idx} (True cx={true_cx:.3f}, cy={true_cy:.3f})...")
 
    z_samples = qz_val.sample(K_samples)[:, sample_idx, :].numpy()
 
    x = np.linspace(0.0, 1.0, N_fem)
    y = np.linspace(0.0, 1.0, N_fem)
    X, Y = np.meshgrid(x, y)
    X_flat_1D = X.flatten()
    Y_flat_1D = Y.flatten()
 
    predictions = []
    for k in range(K_samples):
        cx_k, cy_k = z_samples[k]
        cx_array = np.full_like(X_flat_1D, cx_k)
        cy_array = np.full_like(Y_flat_1D, cy_k)
        X_tensor = tf.convert_to_tensor(
            np.column_stack((X_flat_1D, Y_flat_1D, cx_array, cy_array)), dtype=number_type)
        u_pred = decoder_pinn(X_tensor).numpy().flatten() * U_SCALE * 1000.0
        predictions.append(u_pred)
    predictions = np.array(predictions)
 
    mean_field = np.mean(predictions, axis=0).reshape(N_fem, N_fem)
    std_field = np.std(predictions, axis=0).reshape(N_fem, N_fem)
    relative_std_field = std_field / np.max(np.abs(mean_field))
 
    pred_cx, pred_cy = u_pred_mean[sample_idx]
 
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
 
    v_min, v_max = mean_field.min(), mean_field.max()
    livelli_mean = np.linspace(v_min, v_max, 50)
    c1 = axs[0].contourf(X, Y, mean_field, levels=livelli_mean, cmap='viridis')
    axs[0].set_title('Mean Field Prediction')
    axs[0].set_xlabel('X [cm]')
    axs[0].set_ylabel('Y [cm]')
    fig.colorbar(c1, ax=axs[0], label='Time [ms]')
 
    rect_pred = patches.Rectangle((pred_cx - lato_cicatrice / 2, pred_cy - lato_cicatrice / 2),
                                   lato_cicatrice, lato_cicatrice, linewidth=2,
                                   edgecolor='blue', facecolor='none', linestyle='-')
    rect_true = patches.Rectangle((true_cx - lato_cicatrice / 2, true_cy - lato_cicatrice / 2),
                                   lato_cicatrice, lato_cicatrice, linewidth=2,
                                   edgecolor='red', facecolor='none', linestyle='--')
    axs[0].add_patch(rect_pred)
    axs[0].add_patch(rect_true)
 
    livelli_std = np.linspace(relative_std_field.min(), relative_std_field.max(), 50)
    c2 = axs[1].contourf(X, Y, relative_std_field, levels=livelli_std, cmap='magma')
    axs[1].set_title('Relative Predictive Uncertainty')
    axs[1].set_xlabel('X [cm]')
    axs[1].set_ylabel('Y [cm]')
    fig.colorbar(c2, ax=axs[1], label='Std Dev / Max|Mean|')
 
    rect_pred2 = patches.Rectangle((pred_cx - lato_cicatrice / 2, pred_cy - lato_cicatrice / 2),
                                    lato_cicatrice, lato_cicatrice, linewidth=2,
                                    edgecolor='blue', facecolor='none', linestyle='-')
    axs[1].add_patch(rect_pred2)
 
    plt.tight_layout()
    plt.show()
 
    return mean_field, std_field, relative_std_field
