import numpy as np
from src.physics import constants as c
import tensorflow as tf

def generate_interior_grid(N, split_scar=0.2, dtype=tf.float32):
    """
    Generates the interior of the grid, with oversampling in the scar region if split_scar > 0.

    Args:
        N: number of points to sample
        split_scar: % of points that will be sampled in the scar region (with a 10% outside margin from the scar boundary)
    """
    N_focus = int(N * split_scar)
    N_healthy = N - N_focus

    # Punti su tutto il quadrato 
    uniform_points = np.random.uniform(0.0, 1.0, (N_healthy, 2))

    # Generazione punti sulla cicatrice 
    if N_focus > 0:
        x_min, x_max = c.SCAR_X_MIN, c.SCAR_X_MAX
        y_min, y_max = c.SCAR_Y_MIN, c.SCAR_Y_MAX

        margin = (x_max - x_min) / 10
        scar_x = np.clip(np.random.uniform(x_min - margin, x_max + margin, (N_focus, 1)), 0.0, 1.0)
        scar_y = np.clip(np.random.uniform(y_min - margin, y_max + margin, (N_focus, 1)), 0.0, 1.0)
        scar_points = np.hstack((scar_x, scar_y))
        
        all_points = np.vstack((uniform_points, scar_points))
    else:
        all_points = uniform_points
    
    np.random.shuffle(all_points) 
    all_points = all_points[:N, :]
    return tf.convert_to_tensor(all_points, dtype=dtype)

def generate_boundary_grid(N_activation, N_neumann, r=c.R_A_VAL, dtype=tf.float32):
    """
    Generates the boundary of the grid.

    Args:
        N_activation: number of points to sample in the activation region (Dirichlet BCs)
        N_neumann: number of points to sample per edge (Neumann BCs)
        r: radius of the activation region 
    """
    # DIRICHLET BOUNDARY [0,r] sugli assi
    N_half = N_activation // 2
    
    # Segmento sull'asse x
    x_dir_x = np.random.uniform(0.0, r, N_half)
    y_dir_x = np.zeros_like(x_dir_x)
    
    # Segmento sull'asse y
    x_dir_y = np.zeros_like(x_dir_x)
    y_dir_y = np.random.uniform(0.0, r, N_activation - N_half) 
    
    x_dir = np.concatenate([x_dir_x, x_dir_y])
    y_dir = np.concatenate([y_dir_x, y_dir_y])
    x_dirichlet = np.column_stack((x_dir, y_dir))
    n_dirichlet = np.zeros_like(x_dirichlet) 
    
    # NEUMANN BOUNDARIES 
    # Bottom (y=0, da r in poi), n=(0,-1)
    x_neu_b = np.random.uniform(r, 1, N_neumann)
    pts_neu_b = np.column_stack((x_neu_b, np.zeros_like(x_neu_b)))
    n_neu_b = np.column_stack((np.zeros_like(x_neu_b), -np.ones_like(x_neu_b)))
    
    # Right (x=1), n=(1,0)
    y_neu_r = np.random.uniform(0, 1, N_neumann)
    pts_neu_r = np.column_stack((np.ones_like(y_neu_r), y_neu_r))
    n_neu_r = np.column_stack((np.ones_like(y_neu_r), np.zeros_like(y_neu_r)))
    
    # Top (y=1), n=(0,1)
    x_neu_t = np.random.uniform(0, 1, N_neumann)
    pts_neu_t = np.column_stack((x_neu_t, np.ones_like(x_neu_t)))
    n_neu_t = np.column_stack((np.zeros_like(x_neu_t), np.ones_like(x_neu_t)))
    
    # Left (x=0, da r in poi), n=(-1,0) 
    y_neu_l = np.random.uniform(r, 1, N_neumann)
    pts_neu_l = np.column_stack((np.zeros_like(y_neu_l), y_neu_l))
    n_neu_l = np.column_stack((-np.ones_like(y_neu_l), np.zeros_like(y_neu_l)))
    
    x_neumann = np.vstack((pts_neu_b, pts_neu_r, pts_neu_t, pts_neu_l))
    n_neumann = np.vstack((n_neu_b, n_neu_r, n_neu_t, n_neu_l))
    
    x_boundary = np.vstack((x_neumann, x_dirichlet))
    n_boundary = np.vstack((n_neumann, n_dirichlet))
    
    return tf.convert_to_tensor(x_boundary, dtype=dtype), tf.convert_to_tensor(n_boundary, dtype=dtype)


def generate_parametric_collocation_points(N_interior, N_boundary, N_activation, dtype= tf.float32):
    """
    Generates 4d collocation points [x, y, cx, cy]
    """
    # 2d grid: [x, y]
    x_int_2D = generate_interior_grid(N=N_interior, split_scar=0.0, dtype=dtype)
    x_boundary_2D, n_boundary_2D = generate_boundary_grid(N_activation=N_activation, N_neumann=N_boundary, dtype=dtype)

    # (cx, cy) in [0.2, 0.8]
    c_int = tf.random.uniform((tf.shape(x_int_2D)[0], 2), minval=0.2, maxval=0.8, dtype=dtype)
    c_bound = tf.random.uniform((tf.shape(x_boundary_2D)[0], 2), minval=0.2, maxval=0.8, dtype=dtype)

    X_interior_4D = tf.concat([x_int_2D, c_int], axis=1)
    X_boundary_4D = tf.concat([x_boundary_2D, c_bound], axis=1)

    N_neumann = 4 * N_boundary

    return X_interior_4D, X_boundary_4D, n_boundary_2D, N_neumann 