import tensorflow as tf
from src.physics import constants as c


# PINN CLASS
class PINN(tf.keras.Model):
    # PINN constructor
    def __init__(self, n_neurons, u_scale):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(l, activation='tanh') for l in n_neurons]
        self.out = tf.keras.layers.Dense(1, activation='softplus')

        self.u_scale = u_scale

    def call(self, x):
        z = x
        for layer in self.hidden:
            z = layer(z)
        return self.out(z)


# PDE residual
def compute_residual(model, inputs, M_func):
    """
    computes PDE residual: res = sqrt(M|∇u|²) - (ε * M / C0)Δu - 1/C0

    Args:
        inputs: tensor of shape (N, 4) -> [x, y, cx, cy]
        M_func: callable function returning the diffusion tensor evaluated in x

    Returns: tensor of shape (N, 1) with the PDE residual at each collocation point
    """
    current_dtype = inputs.dtype

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(inputs)
        with tf.GradientTape() as tape1:
            tape1.watch(inputs)
            u_pred = tf.cast(model(inputs), current_dtype)
            scale = tf.cast(model.u_scale, current_dtype)
            u = u_pred * scale

        
        grad_u = tape1.gradient(u, inputs)
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        
    grad_u_x = tape2.gradient(u_x, inputs)
    grad_u_y = tape2.gradient(u_y, inputs)
    del tape2

    u_xx = grad_u_x[:, 0:1]
    u_yy = grad_u_y[:, 1:2]

    x1 = inputs[:, 0:1]
    x2 = inputs[:, 1:2]
    
   # Equation Parameters
    C0 = tf.constant(c.C0_VAL, dtype=current_dtype)
    EPS = tf.constant(c.EPS_VAL, dtype=current_dtype)
    M = M_func(inputs)

    grad_norm = tf.square(u_x) + tf.square(u_y)
    laplacian = u_xx + u_yy

    eikonal = tf.sqrt(M * grad_norm)
    diffusion = (EPS * M / C0) * laplacian
    rhs = tf.constant(1.0, dtype=current_dtype) / C0

    return eikonal - diffusion - rhs


# NEUMANN BC RESIDUAL
def compute_neumann_residual(model, inputs_bc, n_bc):
    """
    computes Neumann BC residual ∇u⋅n = ∂u/∂x1 * n1 + ∂u/∂x2 * n2

    Args:
        inputs_bc: tensor of shape (N, 4) -> [x, y, cx, cy]
        n_bc: normal vector (N, 2)
        n_bc: normal vector (N, 2)

    Returns: tensor of shape (N, 1) with the Neumann residual at each boundary point
    """
    with tf.GradientTape() as tape:
        tape.watch(inputs_bc)
        u = model(inputs_bc) * model.u_scale
    
    grad_u = tape.gradient(u, inputs_bc)
    
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    
    n1 = n_bc[:, 0:1]
    n2 = n_bc[:, 1:2]
    
    # ∇u ⋅ n = 0  
    return u_x * n1 + u_y * n2