import matplotlib.pyplot as plt
import numpy as np

def plot_adam_history(history_adam, save_freq=10):
    """
    Generates a two-panel plot for the Adam optimization history:
    1. The decay of individual physics and boundary loss components.
    2. A direct comparison between Train Data Loss and Validation Data Loss.
    """
    loss_pde_comb = history_adam['pde'] 
    loss_dir_comb = history_adam['dir'] 
    loss_neu_comb = history_adam['neu'] 
    loss_data_comb = history_adam['data'] 
    loss_val_comb = history_adam['val'] 

    # Generate x-axis values based on saving frequency
    x_epochs = np.arange(0, len(loss_pde_comb) * save_freq, save_freq)

    # Plot Loss history
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))

    # Decay of PINN Losses
    ax1.plot(x_epochs, loss_pde_comb, label='PDE Residual', alpha=0.9, color='#1f77b4', linewidth=1.5)
    ax1.plot(x_epochs, loss_dir_comb, label='Dirichlet BC', alpha=0.9, color='#2ca02c', linewidth=1.5)
    ax1.plot(x_epochs, loss_neu_comb, label='Neumann BC', alpha=0.9, color='#9467bd', linewidth=1.5)
    ax1.plot(x_epochs, loss_data_comb, label='FEM Data', alpha=0.9, color='#ff7f0e', linewidth=1.5)

    ax1.set_yscale('log')
    ax1.set_title('Decay of Individual Loss Components (Adam)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Train Data vs Validation Data
    ax2.plot(x_epochs, loss_data_comb, label='Train Data Loss', alpha=0.4, color='#ff7f0e', linestyle='--', linewidth=2)
    ax2.plot(x_epochs, loss_val_comb, label='Validation Data Loss', alpha=0.9, color='#ff7f0e', linewidth=2)

    ax2.set_yscale('log')
    ax2.set_title('Train vs Validation Data Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def clean_lbfgs_history(history_lbfgs):

    cleaned = {k: [] for k in history_lbfgs.keys()}
    last_valid_idx = 0
    current_min_total = float('inf')
    
    for i, current_total in enumerate(history_lbfgs['total']):
        if current_total <= current_min_total:
            current_min_total = current_total 
            last_valid_idx = i
            
        for key in history_lbfgs.keys():
            cleaned[key].append(history_lbfgs[key][last_valid_idx])
            
    return cleaned


def plot_lbfgs_history(history_lbfgs):

    # 1. Clean history
    history_lbfgs_clean = clean_lbfgs_history(history_lbfgs)

    loss_pde_comb = history_lbfgs_clean['pde']
    loss_dir_comb = history_lbfgs_clean['dir']
    loss_neu_comb = history_lbfgs_clean['neu']
    loss_data_comb = history_lbfgs_clean['data']
    loss_val_comb = history_lbfgs_clean['val']

    # 2. Plot Loss history
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))

    # Decay PINN Losses
    ax1.plot(loss_pde_comb, label='PDE Residual', alpha=0.9, color='#1f77b4', linewidth=1.5)
    ax1.plot(loss_dir_comb, label='Dirichlet BC', alpha=0.9, color='#2ca02c', linewidth=1.5)
    ax1.plot(loss_neu_comb, label='Neumann BC', alpha=0.9, color='#9467bd', linewidth=1.5)
    ax1.plot(loss_data_comb, label='FEM Data', alpha=0.9, color='#ff7f0e', linewidth=1.5)

    ax1.set_yscale('log')
    ax1.set_title('Decay of Individual Loss Components (L-BFGS)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Train Data vs Validation Data
    ax2.plot(loss_data_comb, label='Train Data Loss', alpha=0.4, color='#ff7f0e', linestyle='--', linewidth=2)
    ax2.plot(loss_val_comb, label='Validation Data Loss', alpha=0.9, color='#ff7f0e', linewidth=2)

    ax2.set_yscale('log')
    ax2.set_title('Train vs Validation Data Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Function Evaluations', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    


def plot_encoder_adam_history(history, save_freq=10):
    x_epochs = np.arange(0, len(history['total']) * save_freq, save_freq)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))

    ax1.plot(x_epochs, history['total'], label='Train Loss', alpha=0.9, color='#1f77b4', linewidth=1.5)
    ax1.plot(x_epochs, history['val'], label='Validation Loss', alpha=0.9, color='#2ca02c', linewidth=1.5)
    ax1.set_yscale('log'); ax1.set_title('Decay of Loss'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(x_epochs, history['data_fid'], label='Data Fidelity', alpha=0.9, color='#9467bd', linewidth=1.5)
    ax2.plot(x_epochs, history['val_dfid'], label='Validation Data Fidelity', alpha=0.9, color='#ff7f0e', linewidth=1.5)
    ax2.set_yscale('log'); ax2.set_title('Data Fidelity - Reconstruction Error')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()