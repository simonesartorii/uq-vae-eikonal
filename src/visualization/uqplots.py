import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import qmc, norm
from src.visualization.pdeplots import N_GRID
from matplotlib.ticker import FormatStrFormatter

def plot_uncertainty_maps(decoder_pinn, qz_val, u_pred_mean, u_true, sample_indices,
                             U_SCALE, K_samples=8192, N_fem=None, scar_edge=0.2,
                             grid_res=200, number_type=tf.float64):
    """
    Plots heatmaps, mean field and relative standard deviation. 
    Optimized with batched inference and Sobol sequences.
    """
    if N_fem is None:
        N_fem = N_GRID
        
    plt.rcParams.update({
        'font.size': 12,          
        'axes.labelsize': 14,     
        'axes.titlesize': 16,     
        'legend.fontsize': 12
    })
    
    fmt_assi = FormatStrFormatter('%.2g')

    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    u_x = np.linspace(0.0, 1.0, grid_res)
    u_y = np.linspace(0.0, 1.0, grid_res)
    U_X, U_Y = np.meshgrid(u_x, u_y)
    U_grid = np.dstack((U_X, U_Y))
    dU = (u_x[1] - u_x[0]) * (u_y[1] - u_y[0])

    cmap = plt.get_cmap('Blues')

    for i, sample_idx in enumerate(sample_indices):
        ax_heat = axes[i, 0]
        ax_mean = axes[i, 1]
        ax_rel_std = axes[i, 2]

        true_xy = u_true[sample_idx]
        true_cx, true_cy = true_xy.numpy() if hasattr(true_xy, 'numpy') else true_xy
        print(f"Uncertainty Propagation (Sobol) for sample {sample_idx} (True cx={true_cx:.3f}, cy={true_cy:.3f})...")

        mean_z = qz_val.mean()[sample_idx].numpy()
        cov_z = qz_val.covariance()[sample_idx].numpy()
        pred_cx, pred_cy = u_pred_mean[sample_idx]
        
        mvn = tfp.distributions.MultivariateNormalTriL(loc=mean_z, scale_tril=tf.linalg.cholesky(cov_z))
        pdf_U = mvn.prob(U_grid).numpy()

        pdf_flat = pdf_U.flatten()
        sorted_pdf = np.sort(pdf_flat)[::-1]
        cumulative_mass = np.cumsum(sorted_pdf) * dU
        idx_95 = np.argmax(cumulative_mass >= 0.95)
        threshold_95 = sorted_pdf[idx_95]

        ax_heat.set_aspect('equal')
        c = ax_heat.contourf(U_X, U_Y, pdf_U, levels=50, cmap=cmap)

        divider = make_axes_locatable(ax_heat)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(c, cax=cax, format='%.2g')
        cb.set_label('Density', fontsize=12)
        cb.ax.tick_params(labelsize=10)

        ax_heat.contour(U_X, U_Y, pdf_U, levels=[threshold_95], colors='darkorange', linewidths=2, zorder=4)
        ax_heat.scatter(pred_cx, pred_cy, color='white', marker='o', s=60, edgecolors='black', zorder=5)
        ax_heat.scatter(true_cx, true_cy, color='crimson', marker='x', s=80, linewidth=2.5, zorder=5)

        ax_heat.set_xlim(0, 1)
        ax_heat.set_ylim(0, 1)
        
        ax_heat.set_xlabel('cx')
        ax_heat.set_ylabel(f'Sample {sample_idx}\ncy', fontweight='bold')
        ax_heat.xaxis.set_major_formatter(fmt_assi)
        ax_heat.yaxis.set_major_formatter(fmt_assi)
        
        if i == 0:
            ax_heat.set_title('Posterior Distribution')

        sampler = qmc.Sobol(d=2, scramble=True)
        sobol_uniform = sampler.random(n=K_samples)
        sobol_norm_std = norm.ppf(sobol_uniform)
        
        L = np.linalg.cholesky(cov_z)
        z_samples = mean_z + (sobol_norm_std @ L.T)

        x = np.linspace(0.0, 1.0, N_fem)
        y = np.linspace(0.0, 1.0, N_fem)
        X, Y = np.meshgrid(x, y)
        X_flat_1D = X.flatten()
        Y_flat_1D = Y.flatten()
        X_grid = np.column_stack((X_flat_1D, Y_flat_1D))
        N_points = N_fem * N_fem

        predictions = np.zeros((K_samples, N_points))
        BATCH_SIZE = 100 
        for b in range(0, K_samples, BATCH_SIZE):
            end_idx = min(b + BATCH_SIZE, K_samples)
            current_batch_size = end_idx - b
            
            z_batch = z_samples[b:end_idx] 
            
            X_tiled = np.tile(X_grid, (current_batch_size, 1))
            Z_repeated = np.repeat(z_batch, N_points, axis=0)
            
            input_batch = np.column_stack((X_tiled, Z_repeated))
            input_tensor = tf.convert_to_tensor(input_batch, dtype=number_type)
            
            u_pred_batch = decoder_pinn(input_tensor).numpy().flatten() * U_SCALE * 1000.0
            predictions[b:end_idx, :] = u_pred_batch.reshape(current_batch_size, N_points)

        mean_field = np.mean(predictions, axis=0).reshape(N_fem, N_fem)
        var_field = np.var(predictions, axis=0).reshape(N_fem, N_fem)
        std_field = np.sqrt(var_field)
        relative_std_field = std_field / (np.max(np.abs(mean_field)) + 1e-8)

        # Plot Mean Field
        v_min, v_max = mean_field.min(), mean_field.max()
        livelli_mean = np.linspace(v_min, v_max, 50)
        c1 = ax_mean.contourf(X, Y, mean_field, levels=livelli_mean, cmap='viridis')
        if i == 0:
            ax_mean.set_title('Mean Field')
        ax_mean.set_xlabel('X [cm]')
        ax_mean.set_ylabel('Y [cm]')
        ax_mean.xaxis.set_major_formatter(fmt_assi)
        ax_mean.yaxis.set_major_formatter(fmt_assi)
        cb1 = fig.colorbar(c1, ax=ax_mean, format='%.2g')
        cb1.set_label('Time [ms]', fontsize=12)
        cb1.ax.tick_params(labelsize=10)

        rect_pred = patches.Rectangle((pred_cx - scar_edge / 2, pred_cy - scar_edge / 2),
                                      scar_edge, scar_edge, linewidth=2,
                                      edgecolor='blue', facecolor='none', linestyle='-')
        rect_true = patches.Rectangle((true_cx - scar_edge / 2, true_cy - scar_edge / 2),
                                      scar_edge, scar_edge, linewidth=2,
                                      edgecolor='red', facecolor='none', linestyle='--')
        ax_mean.add_patch(rect_pred)
        ax_mean.add_patch(rect_true)

        livelli_rel_std = np.linspace(relative_std_field.min(), relative_std_field.max(), 50)
        
        c3 = ax_rel_std.contourf(X, Y, relative_std_field, levels=livelli_rel_std, cmap='magma')
        
        if i == 0:
            ax_rel_std.set_title('Relative Standard Deviation')
        ax_rel_std.set_xlabel('X [cm]')
        ax_rel_std.set_ylabel('Y [cm]')
        ax_rel_std.xaxis.set_major_formatter(fmt_assi)
        ax_rel_std.yaxis.set_major_formatter(fmt_assi)

        cb3 = fig.colorbar(c3, ax=ax_rel_std, format='%.2g')
        cb3.set_label('Std / Max|Mean|', fontsize=12)
        cb3.ax.tick_params(labelsize=10)
        legend_elements = [
        Line2D([0], [0], color='darkorange', lw=2.5, label='95% Confidence Region'),
        Line2D([0], [0], marker='o', color='w', label='MAP', markerfacecolor='white', markeredgecolor='black', markersize=9),
        Line2D([0], [0], marker='x', color='w', label='True Scar Center', markeredgecolor='crimson', markersize=10, markeredgewidth=2.5)
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=11, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig('results/pipeline_evolution/04_uq-vae/posterior_maps.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

def propagate_and_plot_uncertainty_on_axes_sobol(decoder_pinn, qz_val, u_pred_mean, u_true, sample_idx,
                                           axes, U_SCALE, K_samples=1000, N_fem=None, scar_edge=0.2,
                                           number_type=tf.float64):
    """
    Plots mean field, variance, and relative predictive uncertainty 
    on a tuple of 3 matplotlib axes. Optimized with batched inference and Sobol sequences.
    """
    if N_fem is None:
        N_fem = N_GRID
        
    ax_mean, ax_var, ax_rel_var = axes
    fig = ax_mean.figure

    true_xy = u_true[sample_idx]
    true_cx, true_cy = true_xy.numpy() if hasattr(true_xy, 'numpy') else true_xy
    print(f"Uncertainty Propagation (Sobol) for sample {sample_idx} (True cx={true_cx:.3f}, cy={true_cy:.3f})...")

    mean_z = qz_val.mean()[sample_idx].numpy()
    cov_z = qz_val.covariance()[sample_idx].numpy()
    
    sampler = qmc.Sobol(d=2, scramble=True)
    sobol_uniform = sampler.random(n=K_samples)
    sobol_norm_std = norm.ppf(sobol_uniform)
    
    L = np.linalg.cholesky(cov_z)
    z_samples = mean_z + (sobol_norm_std @ L.T)

    x = np.linspace(0.0, 1.0, N_fem)
    y = np.linspace(0.0, 1.0, N_fem)
    X, Y = np.meshgrid(x, y)
    X_flat_1D = X.flatten()
    Y_flat_1D = Y.flatten()
    X_grid = np.column_stack((X_flat_1D, Y_flat_1D))
    N_points = N_fem * N_fem

    predictions = np.zeros((K_samples, N_points))
    BATCH_SIZE = 100 
    for i in range(0, K_samples, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, K_samples)
        current_batch_size = end_idx - i
        
        z_batch = z_samples[i:end_idx] 
        
        X_tiled = np.tile(X_grid, (current_batch_size, 1))
        Z_repeated = np.repeat(z_batch, N_points, axis=0)
        
        input_batch = np.column_stack((X_tiled, Z_repeated))
        input_tensor = tf.convert_to_tensor(input_batch, dtype=number_type)
        
        u_pred_batch = decoder_pinn(input_tensor).numpy().flatten() * U_SCALE * 1000.0
        predictions[i:end_idx, :] = u_pred_batch.reshape(current_batch_size, N_points)

    mean_field = np.mean(predictions, axis=0).reshape(N_fem, N_fem)
    var_field = np.var(predictions, axis=0).reshape(N_fem, N_fem)
    relative_var_field = var_field / (np.max(np.abs(mean_field)) + 1e-8)

    pred_cx, pred_cy = u_pred_mean[sample_idx]

    # Plot Mean Field
    v_min, v_max = mean_field.min(), mean_field.max()
    livelli_mean = np.linspace(v_min, v_max, 50)
    c1 = ax_mean.contourf(X, Y, mean_field, levels=livelli_mean, cmap='viridis')
    ax_mean.set_title(f'Sample {sample_idx}: Mean Field (Sobol n={K_samples})')
    ax_mean.set_xlabel('X [cm]')
    ax_mean.set_ylabel('Y [cm]')
    fig.colorbar(c1, ax=ax_mean, label='Time [ms]')

    rect_pred = patches.Rectangle((pred_cx - scar_edge / 2, pred_cy - scar_edge / 2),
                                  scar_edge, scar_edge, linewidth=2,
                                  edgecolor='blue', facecolor='none', linestyle='-')
    rect_true = patches.Rectangle((true_cx - scar_edge / 2, true_cy - scar_edge / 2),
                                  scar_edge, scar_edge, linewidth=2,
                                  edgecolor='red', facecolor='none', linestyle='--')
    ax_mean.add_patch(rect_pred)
    ax_mean.add_patch(rect_true)

    # Plot Variance Field
    livelli_var = np.linspace(var_field.min(), var_field.max(), 50)
    c2 = ax_var.contourf(X, Y, var_field, levels=livelli_var, cmap='plasma')
    ax_var.set_title('Predictive Uncertainty (Variance)')
    ax_var.set_xlabel('X [cm]')
    ax_var.set_ylabel('Y [cm]')
    fig.colorbar(c2, ax=ax_var, label='Variance [ms²]')

    rect_pred2 = patches.Rectangle((pred_cx - scar_edge / 2, pred_cy - scar_edge / 2),
                                   scar_edge, scar_edge, linewidth=2,
                                   edgecolor='blue', facecolor='none', linestyle='-')
    ax_var.add_patch(rect_pred2)

    # Plot Relative Variance Field 
    livelli_rel_var = np.linspace(relative_var_field.min(), relative_var_field.max(), 50)
    c3 = ax_rel_var.contourf(X, Y, relative_var_field, levels=livelli_rel_var, cmap='magma')
    ax_rel_var.set_title('Relative Predictive Uncertainty')
    ax_rel_var.set_xlabel('X [cm]')
    ax_rel_var.set_ylabel('Y [cm]')
    fig.colorbar(c3, ax=ax_rel_var, label='Variance / Max|Mean|')

    rect_pred3 = patches.Rectangle((pred_cx - scar_edge / 2, pred_cy - scar_edge / 2),
                                   scar_edge, scar_edge, linewidth=2,
                                   edgecolor='blue', facecolor='none', linestyle='-')
    ax_rel_var.add_patch(rect_pred3)

    return mean_field, var_field, relative_var_field


def plot_multiple_posterior_heatmaps(qz_val, u_true, sample_indices, grid_res=200):
    """
    Plots a 2x3 grid of posterior distribution heatmaps in the physical domain [0,1]x[0,1]
    with the 95% confidence interval regions.
    
    Args:
        qz_val: distribution returned by the encoder for the validation set 
        u_true: true [cx, cy] for all validation samples (N,2)
        sample_indices: list or array of exactly 6 sample indices to plot
        grid_res: resolution of the heatmap grid
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    u_x = np.linspace(0.0, 1.0, grid_res)
    u_y = np.linspace(0.0, 1.0, grid_res)
    U_X, U_Y = np.meshgrid(u_x, u_y)
    U_grid = np.dstack((U_X, U_Y))
    dU = (u_x[1] - u_x[0]) * (u_y[1] - u_y[0])

    cmap = plt.get_cmap('Blues')

    for i, (sample_idx, ax) in enumerate(zip(sample_indices, axes.flatten())):
        true_xy = u_true[sample_idx]
        true_cx, true_cy = true_xy.numpy() if hasattr(true_xy, 'numpy') else true_xy
        
        mean_u = qz_val.mean()[sample_idx].numpy()
        cov_u = qz_val.covariance()[sample_idx].numpy()
        pred_cx, pred_cy = mean_u

        mvn = tfp.distributions.MultivariateNormalTriL(loc=mean_u, scale_tril=tf.linalg.cholesky(cov_u))
        pdf_U = mvn.prob(U_grid).numpy()

        pdf_flat = pdf_U.flatten()
        sorted_pdf = np.sort(pdf_flat)[::-1]
        cumulative_mass = np.cumsum(sorted_pdf) * dU
        idx_95 = np.argmax(cumulative_mass >= 0.95)
        threshold_95 = sorted_pdf[idx_95]

        ax.set_aspect('equal')
        c = ax.contourf(U_X, U_Y, pdf_U, levels=50, cmap=cmap)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(c, cax=cax)
        cb.set_label('Density', fontsize=9)
        cb.ax.tick_params(labelsize=8)

        ax.contour(U_X, U_Y, pdf_U, levels=[threshold_95], colors='darkorange', linewidths=2, zorder=4)
        ax.scatter(pred_cx, pred_cy, color='white', marker='o', s=60, edgecolors='black', zorder=5)
        ax.scatter(true_cx, true_cy, color='crimson', marker='x', s=80, linewidth=2.5, zorder=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Sample {sample_idx}', fontsize=12)
        
        if i >= 3:
            ax.set_xlabel('cx', fontsize=10)
        if i % 3 == 0:
            ax.set_ylabel('cy', fontsize=10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkorange', lw=2.5, label='95% Confidence Region'),
        Line2D([0], [0], marker='o', color='w', label='Posterior Mean', markerfacecolor='white', markeredgecolor='black', markersize=9),
        Line2D([0], [0], marker='x', color='w', label='True Scar Center', markeredgecolor='crimson', markersize=10, markeredgewidth=2.5)
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=11, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=2.0, h_pad=2.0)
