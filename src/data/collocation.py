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
    uniform_points = tf.random.uniform((N_healthy, 2), minval=0.0, maxval=1.0, dtype=dtype)

    # Generazione punti sulla cicatrice
    if N_focus > 0:
        x_min = tf.cast(c.SCAR_X_MIN)
        x_max = tf.cast(c.SCAR_X_MAX)
        y_min = tf.cast(c.SCAR_Y_MIN)
        y_max = tf.cast(c.SCAR_Y_MAX)

        margin = (x_max - x_min) / 10.0
        scar_x = tf.random.uniform((N_focus, 1), minval=x_min - margin, maxval=x_max + margin, dtype=dtype)
        scar_y = tf.random.uniform((N_focus, 1), minval=y_min - margin, maxval=y_max + margin, dtype=dtype)
        scar_x = tf.clip_by_value(scar_x, 0.0, 1.0)
        scar_y = tf.clip_by_value(scar_y, 0.0, 1.0)
        scar_points = tf.concat([scar_x, scar_y], axis=1)

        all_points = tf.concat([uniform_points, scar_points], axis=0)
    else:
        all_points = uniform_points

    all_points = all_points[:N, :]
    return all_points

def generate_boundary_grid(N_activation, N_neumann, r=c.R_A_VAL, dtype=tf.float32):
    """
    Generates the boundary of the grid.

    Args:
        N_activation: number of points to sample in the activation region (Dirichlet BCs)
        N_neumann: number of points to sample per edge (Neumann BCs)
        r: radius of the activation region
    """
    r = tf.cast(r, dtype)

    # DIRICHLET BOUNDARY [0,r] sugli assi
    N_half = N_activation // 2
    N_rest = N_activation - N_half

    # Segmento sull'asse x
    x_dir_x = tf.random.uniform((N_half,), minval=0.0, maxval=r, dtype=dtype)
    y_dir_x = tf.zeros_like(x_dir_x)

    # Segmento sull'asse y
    x_dir_y = tf.zeros((N_rest,), dtype=dtype)
    y_dir_y = tf.random.uniform((N_rest,), minval=0.0, maxval=r, dtype=dtype)

    x_dir = tf.concat([x_dir_x, x_dir_y], axis=0)
    y_dir = tf.concat([y_dir_x, y_dir_y], axis=0)
    x_dirichlet = tf.stack([x_dir, y_dir], axis=1)
    n_dirichlet = tf.zeros_like(x_dirichlet)

    # NEUMANN BOUNDARIES
    # Bottom (y=0, da r in poi), n=(0,-1)
    x_neu_b = tf.random.uniform((N_neumann,), minval=r, maxval=1.0, dtype=dtype)
    pts_neu_b = tf.stack([x_neu_b, tf.zeros_like(x_neu_b)], axis=1)
    n_neu_b = tf.stack([tf.zeros_like(x_neu_b), -tf.ones_like(x_neu_b)], axis=1)

    # Right (x=1), n=(1,0)
    y_neu_r = tf.random.uniform((N_neumann,), minval=0.0, maxval=1.0, dtype=dtype)
    pts_neu_r = tf.stack([tf.ones_like(y_neu_r), y_neu_r], axis=1)
    n_neu_r = tf.stack([tf.ones_like(y_neu_r), tf.zeros_like(y_neu_r)], axis=1)

    # Top (y=1), n=(0,1)
    x_neu_t = tf.random.uniform((N_neumann,), minval=0.0, maxval=1.0, dtype=dtype)
    pts_neu_t = tf.stack([x_neu_t, tf.ones_like(x_neu_t)], axis=1)
    n_neu_t = tf.stack([tf.zeros_like(x_neu_t), tf.ones_like(x_neu_t)], axis=1)

    # Left (x=0, da r in poi), n=(-1,0)
    y_neu_l = tf.random.uniform((N_neumann,), minval=r, maxval=1.0, dtype=dtype)
    pts_neu_l = tf.stack([tf.zeros_like(y_neu_l), y_neu_l], axis=1)
    n_neu_l = tf.stack([-tf.ones_like(y_neu_l), tf.zeros_like(y_neu_l)], axis=1)

    x_neumann = tf.concat([pts_neu_b, pts_neu_r, pts_neu_t, pts_neu_l], axis=0)
    n_neumann = tf.concat([n_neu_b, n_neu_r, n_neu_t, n_neu_l], axis=0)

    x_boundary = tf.concat([x_neumann, x_dirichlet], axis=0)
    n_boundary = tf.concat([n_neumann, n_dirichlet], axis=0)

    return x_boundary, n_boundary

@tf.function
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