import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import src.physics.constants as c
import matplotlib.patches as patches

N_GRID = 100


def get_eval_grid(N, cx=None, cy=None, dtype=tf.float32):
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y)
    
    X_flat_1D = X.flatten()
    Y_flat_1D = Y.flatten()

    if cx is not None and cy is not None: # parametric 
        cx_array = np.full_like(X_flat_1D, cx)
        cy_array = np.full_like(Y_flat_1D, cy)
        X_flat = np.column_stack((X_flat_1D, Y_flat_1D, cx_array, cy_array))
    else: # non parametric 
        X_flat = np.column_stack((X_flat_1D, Y_flat_1D))

    X_tensor = tf.convert_to_tensor(X_flat, dtype=dtype)
    
    return X, Y, X_tensor


def plot_residual_map(X_res,Y_res,residual_grid, cx_test=None, cy_test=None, scar_edge=None):
    plt.figure(figsize=(8, 6))
    mappa_res = plt.contourf(X_res, Y_res, residual_grid, levels=50, cmap='magma')
    plt.colorbar(mappa_res, label="|Residuo PDE| (Adimensionale)")

    if cx_test is None:
        cx_test = (c.SCAR_X_MAX + c.SCAR_X_MIN) / 2.0
    if cy_test is None:
        cy_test = (c.SCAR_Y_MAX + c.SCAR_Y_MIN) / 2.0
    if scar_edge is None:
        scar_edge = c.SCAR_X_MAX - c.SCAR_X_MIN
    # Highlight scar
    rect = patches.Rectangle(
        (cx_test - scar_edge/2, cy_test - scar_edge/2), 
        scar_edge, scar_edge,                            
        linewidth=1, edgecolor='green', facecolor='none', linestyle='--'
    )
    plt.gca().add_patch(rect)

    plt.title(f"Mappa del Residuo Fisico (Violazione dell'Equazione) - cx={cx_test}, cy={cy_test}")
    plt.xlabel("X [cm]")
    plt.ylabel("Y [cm]")
    plt.show()

def plot_fixed_scar_comparison(X_fem, Y_fem, U_pinn2D, U_fem2D, Rel_err2D):
    v_max_err = Rel_err2D.max()
    # sanity check to avoid division by zero
    if v_max_err == 0: 
        v_max_err = 1e-8
        
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # min and max values for the color scale
    v_min = min(U_pinn2D.min(), U_fem2D.min())
    v_max = max(U_pinn2D.max(), U_fem2D.max())

    livelli = np.linspace(v_min, v_max, 50)
    livelli_err = np.linspace(0.0, v_max_err, 50)

    # 1. PINN SOLUTION
    c1 = axs[0].contourf(X_fem, Y_fem, U_pinn2D, levels=livelli, cmap='viridis')
    axs[0].set_title('PINN Solution', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('X [cm]')
    axs[0].set_ylabel('Y [cm]')
    fig.colorbar(c1, ax=axs[0], label='Time [ms]')

    # 2. FEM SOLUTION
    c2 = axs[1].contourf(X_fem, Y_fem, U_fem2D, levels=livelli, cmap='viridis')
    axs[1].set_title('FEM Solution', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('X [cm]')
    axs[1].set_ylabel('Y [cm]')
    fig.colorbar(c2, ax=axs[1], label='Time [ms]')

    # 3. RELATIVE ERROR (rimosso extend='max')
    c3 = axs[2].contourf(X_fem, Y_fem, Rel_err2D, levels=livelli_err, cmap='magma', vmin=0.0)
    axs[2].set_title('Relative Error |PINN - FEM| / max(|FEM|)', fontsize=12, fontweight='bold')
    axs[2].set_xlabel('X [cm]')
    axs[2].set_ylabel('Y [cm]')
    fig.colorbar(c3, ax=axs[2], label='Relative Error')

    plt.tight_layout()
    plt.show()


def plot_pinn_solution(N_plot, model, U_SCALE):
    x_val = np.linspace(0, 1, N_plot)
    y_val = np.linspace(0, 1, N_plot)
    X, Y = np.meshgrid(x_val, y_val)
    X_flat = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    u_pred_flat = model(tf.convert_to_tensor(X_flat, dtype=tf.float32)).numpy() * U_SCALE
    U_PINN = u_pred_flat.reshape(N_plot, N_plot)*1000.0

    plt.figure(figsize=(8, 6))
    mappa = plt.contourf(X, Y, U_PINN, levels=50, cmap='viridis')
    plt.colorbar(mappa, label="Time [ms]")
    plt.contour(X, Y, U_PINN, colors="black", linewidths=0.5, levels=20)
    plt.title("PINN Solution")
    plt.xlabel("X [cm]")
    plt.ylabel("Y [cm]")
    plt.show()


def plot_parametric_pinn_solution(X_fem, Y_fem, U_pinn2D, cx_test, cy_test, lato_cicatrice=0.2):
    fig, axs = plt.subplots(1, 1, figsize=(8,6))

    # max and min values
    v_min = U_pinn2D.min()
    v_max = U_pinn2D.max()

    # Gestione del caso in cui v_min == v_max (soluzione piatta)
    if v_min == v_max:
        v_max = v_min + 1e-8

    livelli = np.linspace(v_min, v_max, 50)

    # PINN SOLUTION
    c1 = axs.contourf(X_fem, Y_fem, U_pinn2D, levels=livelli, cmap='viridis')
    axs.set_title(f'PINN Solution (Scar at cx={cx_test}, cy={cy_test})', fontsize=12, fontweight='bold')
    axs.set_xlabel('X [cm]')
    axs.set_ylabel('Y [cm]')
    fig.colorbar(c1, ax=axs, label='Time [ms]')

    axs.contour(X_fem, Y_fem, U_pinn2D, colors="black", linewidths=0.5, levels=20)

    # SCAR
    rect = patches.Rectangle(
        (cx_test - lato_cicatrice/2, cy_test - lato_cicatrice/2), 
        lato_cicatrice, lato_cicatrice,                            
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    axs.add_patch(rect)

    plt.tight_layout()
    plt.show()


def evaluate_and_plot_parametric_scars(model, coords_test, u_fem_all, U_SCALE, 
                                       num_to_plot=5, lato_cicatrice=0.2, dtype=tf.float32):
    """
    Computes the relative L2 error for all scar configurations in the test set.
    Then, it randomly selects a subset of configurations and generates a comparative 
    plot (FEM vs PINN vs Relative Error) for each.
    """
    cx_all = coords_test[:, 2]
    cy_all = coords_test[:, 3]

    # Find all unique (cx, cy) pairs in the test set
    unique_scars = np.unique(np.column_stack((cx_all, cy_all)), axis=0)

    print(f"{len(unique_scars)} different scar configurations in the test set.\n")
    print(f"{'Configuration (cx, cy)':<25} | {'Relative L2 Error':<18}")
    print("-" * 47)

    errori_l2 = []

    # 1. Compute errors for all configurations
    for idx, (cx_val, cy_val) in enumerate(unique_scars):
        mask = (cx_all == cx_val) & (cy_all == cy_val)
        coords_subset = coords_test[mask]
        u_fem_subset = u_fem_all[mask].flatten()
        
        coords_tensor = tf.convert_to_tensor(coords_subset, dtype=dtype)
        u_pinn_subset = model(coords_tensor).numpy().flatten() * U_SCALE
        
        l2_err_scar = np.linalg.norm(u_pinn_subset - u_fem_subset) / np.linalg.norm(u_fem_subset)
        errori_l2.append(l2_err_scar)
        print(f"Scar at cx={cx_val:.2f}, cy={cy_val:.2f}    | {l2_err_scar:.2%}")

    print(f"\nMax relative L2 error: {max(errori_l2):.2%}\n")

    # 2. Random selection of configurations to plot
    num_to_plot = min(num_to_plot, len(unique_scars))
    random_indices = np.random.choice(len(unique_scars), size=num_to_plot, replace=False)
    selected_scars = unique_scars[random_indices]

    fig, axes = plt.subplots(num_to_plot, 3, figsize=(15, 4 * num_to_plot))
    if num_to_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    # 3. Plot generation
    for idx, (cx_val, cy_val) in enumerate(selected_scars):
        mask = (cx_all == cx_val) & (cy_all == cy_val)
        coords_subset = coords_test[mask]
        u_fem_subset = u_fem_all[mask].flatten()
        
        # PINN prediction
        coords_tensor = tf.convert_to_tensor(coords_subset, dtype=dtype)
        u_pinn_subset = model(coords_tensor).numpy().flatten() * U_SCALE
        
        grid_dim = int(np.sqrt(len(u_fem_subset)))
        
        # Extract spatial grid directly from data
        X_grid = coords_subset[:, 0].reshape(grid_dim, grid_dim)
        Y_grid = coords_subset[:, 1].reshape(grid_dim, grid_dim)
            
        u_fem_2d = u_fem_subset.reshape(grid_dim, grid_dim)
        u_pinn_2d = u_pinn_subset.reshape(grid_dim, grid_dim)
        
        # 2D Relative Error
        max_fem = np.max(np.abs(u_fem_2d))
        if max_fem == 0: 
            max_fem = 1e-8
            
        Rel_err2D = np.abs(u_pinn_2d - u_fem_2d) / max_fem
        v_max_err = Rel_err2D.max()
        
        if v_max_err == 0: 
            v_max_err = 1e-8 
            
        livelli_err = np.linspace(0.0, v_max_err, 50)
        
        # FEM
        c0 = axes[idx, 0].contourf(X_grid, Y_grid, u_fem_2d, levels=50, cmap='viridis')
        if idx == 0:
            axes[idx, 0].set_title("Exact Solution (FEM)", fontsize=12, fontweight='bold')
        axes[idx, 0].set_xlabel('X [cm]')
        axes[idx, 0].set_ylabel(f"Config #{idx+1} (cx={cx_val:.2f}, cy={cy_val:.2f})\nY [cm]", fontsize=10, fontweight='bold')
        fig.colorbar(c0, ax=axes[idx, 0], fraction=0.046, pad=0.04)
        
        rect_fem = patches.Rectangle(
            (cx_val - lato_cicatrice/2, cy_val - lato_cicatrice/2), 
            lato_cicatrice, lato_cicatrice,                                     
            linewidth=1, edgecolor='red', facecolor='none', linestyle='--'
        )
        axes[idx, 0].add_patch(rect_fem)
        
        # PINN
        c1 = axes[idx, 1].contourf(X_grid, Y_grid, u_pinn_2d, levels=50, cmap='viridis')
        if idx == 0:
            axes[idx, 1].set_title("PINN Solution", fontsize=12, fontweight='bold')
        axes[idx, 1].set_xlabel('X [cm]')
        axes[idx, 1].set_ylabel('Y [cm]')
        fig.colorbar(c1, ax=axes[idx, 1], fraction=0.046, pad=0.04)
        
        rect_pinn = patches.Rectangle(
            (cx_val - lato_cicatrice/2, cy_val - lato_cicatrice/2), 
            lato_cicatrice, lato_cicatrice,                                     
            linewidth=1, edgecolor='red', facecolor='none', linestyle='--'
        )
        axes[idx, 1].add_patch(rect_pinn)
        
        # RELATIVE ERROR
        c2 = axes[idx, 2].contourf(X_grid, Y_grid, Rel_err2D, levels=livelli_err, cmap='magma', vmin=0.0)
        if idx == 0:
            axes[idx, 2].set_title('Relative Error |PINN - FEM| / max(|FEM|)', fontsize=12, fontweight='bold')
        axes[idx, 2].set_xlabel('X [cm]')
        axes[idx, 2].set_ylabel('Y [cm]')
        fig.colorbar(c2, ax=axes[idx, 2], fraction=0.046, pad=0.04, label='Relative Error')
        
        rect_res = patches.Rectangle(
            (cx_val - lato_cicatrice/2, cy_val - lato_cicatrice/2), 
            lato_cicatrice, lato_cicatrice,                                     
            linewidth=1, edgecolor='green', facecolor='none', linestyle='--'
        )
        axes[idx, 2].add_patch(rect_res)

    plt.tight_layout()
    plt.show()
