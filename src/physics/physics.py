from src.physics import constants as c
import tensorflow as tf

def get_M_const(x):
    """ Returns constant diffusion tensor """
    current_dtype = x.dtype
    m_val = tf.constant(c.M_VAL, dtype=current_dtype)

    return tf.ones((tf.shape(x)[0], 1), dtype=current_dtype) * m_val


def get_M_non_const(x, k=2500.0):
    """ Returns non constant diffusion tensor """
    current_dtype = x.dtype
    k = tf.constant(k, dtype=current_dtype)

    x1 = x[:, 0:1]
    x2 = x[:, 1:2]

    x_min, x_max = c.SCAR_X_MIN, c.SCAR_X_MAX
    y_min, y_max = c.SCAR_Y_MIN, c.SCAR_Y_MAX
    
    M_healthy = tf.constant(c.M_VAL, dtype=current_dtype)
    M_scar = tf.constant(c.M_SCAR, dtype=current_dtype)       
    
    sig_x = tf.math.sigmoid(k * (x1 - x_min)) - tf.math.sigmoid(k * (x1 - x_max))
    sig_y = tf.math.sigmoid(k * (x2 - y_min)) - tf.math.sigmoid(k * (x2 - y_max))
    
    scar_mask = sig_x * sig_y
    
    return M_healthy + scar_mask * (M_scar - M_healthy)
    

def get_M_dynamic(inputs, k=2500.0):
    """ 
    Returns non-constant diffusion tensor dynamically based on scar center.
    inputs: tensor of shape (N, 4) -> [x, y, cx, cy]
    """
    current_dtype = inputs.dtype
    k = tf.constant(k, dtype=current_dtype)
    L_half = tf.constant(0.1, dtype=current_dtype)

    x1 = inputs[:, 0:1]
    x2 = inputs[:, 1:2]
    cx = inputs[:, 2:3]
    cy = inputs[:, 3:4]

    x_min = cx - L_half
    x_max = cx + L_half
    y_min = cy - L_half
    y_max = cy + L_half
    
    M_healthy = tf.constant(c.M_VAL, dtype=current_dtype)
    M_scar = tf.constant(c.M_SCAR, dtype=current_dtype)       
    
    sig_x = tf.math.sigmoid(k * (x1 - x_min)) - tf.math.sigmoid(k * (x1 - x_max))
    sig_y = tf.math.sigmoid(k * (x2 - y_min)) - tf.math.sigmoid(k * (x2 - y_max))
    
    scar_mask = sig_x * sig_y
    
    return M_healthy + scar_mask * (M_scar - M_healthy)