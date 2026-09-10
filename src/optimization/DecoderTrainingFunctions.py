import tensorflow as tf
from src.networks.DecoderPINN import compute_residual, compute_neumann_residual

def loss_fn(model, x_interior, x_boundary, n_boundary, num_neumann, x_data, u_data,
            M_fun, LAMBDA_PDE, LAMBDA_DIR, LAMBDA_NEU, LAMBDA_DATA):
    """
    Args:
        x_interior: (N, 2) tensor containing the coordinates of the interior points
        x_boundary: (N, 2) tensor containing the coordinates of the boundary points
        n_boundary: (N, 2) tensor containing the components of the normal to the boundary for each point in x_boundary
        num_neumann: number of points with Neumann BCs 
        x_data: (N, 2) tensor containing the coordinates of the FEM nodes
        u_data: (N, 1) tensor containing the FEM solution evaluated in x_data
    """
    res_pde = compute_residual(model, x_interior, M_fun)
    loss_pde = tf.reduce_mean(tf.square(res_pde))

    x_neumann = x_boundary[:num_neumann, :]
    n_neumann = n_boundary[:num_neumann, :]
    x_dirichlet = x_boundary[num_neumann:, :]
    
    u_dir = model(x_dirichlet) * model.u_scale
    loss_dir = tf.reduce_mean(tf.square(u_dir))

    res_neu = compute_neumann_residual(model, x_neumann, n_neumann)
    loss_neu = tf.reduce_mean(tf.square(res_neu))

    u_pred_data = model(x_data)
    loss_data = tf.reduce_mean(tf.square(u_pred_data - u_data))

    total_loss = LAMBDA_PDE * loss_pde + LAMBDA_DIR * loss_dir + LAMBDA_NEU * loss_neu + LAMBDA_DATA * loss_data
    return total_loss, loss_pde, loss_dir, loss_neu, loss_data

# TRAIN
def train_step(model, optimizer, x_int, x_boundary, n_boundary, num_neumann, x_data, u_data,
               M_fun, LAMBDA_PDE, LAMBDA_DIR, LAMBDA_NEU, LAMBDA_DATA):
    with tf.GradientTape() as tape:
        loss, lpde, ldir, lneu, ldata = loss_fn(
            model, x_int, x_boundary, n_boundary, num_neumann, x_data, u_data,
            M_fun, LAMBDA_PDE, LAMBDA_DIR, LAMBDA_NEU, LAMBDA_DATA)
        
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, lpde, ldir, lneu, ldata

# VALIDATION
def val_step(model, x_test, u_val):
    u_pred = model(x_test)
    return tf.reduce_mean(tf.square(u_pred - u_val))

# XLA TRAINING and VALIDATION STEPS
@tf.function(jit_compile=True)
def train_step_xla(model, optimizer, x_int, x_boundary, n_boundary, num_neumann, x_data, u_data,
                   M_fun, LAMBDA_PDE, LAMBDA_DIR, LAMBDA_NEU, LAMBDA_DATA):
    """
    XLA-compiled version
    """
    with tf.GradientTape() as tape:
        loss, lpde, ldir, lneu, ldata = loss_fn(
            model, x_int, x_boundary, n_boundary, num_neumann, x_data, u_data,
            M_fun, LAMBDA_PDE, LAMBDA_DIR, LAMBDA_NEU, LAMBDA_DATA)
        
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0) 
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, lpde, ldir, lneu, ldata

@tf.function(jit_compile=True)
def val_step_xla(model, x_test, u_val):
    u_pred = model(x_test)
    return tf.reduce_mean(tf.square(u_pred - u_val))