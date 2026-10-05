import tensorflow as tf

def EncoderLoss(qz, x_obs, y_obs, pinn_model, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, Gamma_E_inv, theta=1e-4):
    """
    Args:
        qz: posterior distribution predicted by the encoder (q_phi(z | y_obs))
        y_obs: observed data (noisy FEM solution evaluated in x_obs)
        x_obs: spatial coordinates where we evaluate y_obs
        pinn_model: decoder (X->U)
        mu_pr: prior mean 
        Gamma_pr: prior covariance matrix (covariance of distribution p(z))
        Gamma_pr_inv: inverse of prior covariance matrix 
        mu_E: error mean (observation noise + decoder error)
        Gamma_E_inv: inverse of error covariance matrix (observation + decoder)
        theta: perturbation parameter for Jacobian trace approximation 

    """
    B = tf.shape(y_obs)[0]
    D_y = tf.shape(y_obs)[1]
    D_z = qz.mean().shape[-1]

    # posterior mean and covariance matrix 
    mu = tf.cast(qz.mean(), dtype=y_obs.dtype)          # (batch_size, D_z)
    Gamma = tf.cast(qz.covariance(), dtype=y_obs.dtype) # (batch_size, D_z, D_z)
    C = tf.cast(qz.scale.to_dense(), dtype=y_obs.dtype) # (batch_size, D_z, D_z) Cholesky factor: Gamma = C C.T

    
    # prior and error matrices bcast
    Gamma_pr_batch = tf.tile(Gamma_pr[tf.newaxis, :, :], [B, 1, 1])
    Gamma_pr_inv_batch = tf.tile(Gamma_pr_inv[tf.newaxis, :, :], [B, 1, 1])
    Gamma_E_inv_batch = tf.tile(Gamma_E_inv[tf.newaxis, :, :], [B, 1, 1])

    # tr(Gamma^-1 * Gamma_pr)
    X1 = tf.linalg.cholesky_solve(C, Gamma_pr_batch) 
    prior_var_1 = (theta**2) * tf.linalg.trace(X1)

    # || mu - mu_pr ||^2_{Gamma_pr^-1}: distance 
    diff_mu = mu - mu_pr 
    diff_mu_weighted = tf.squeeze(tf.matmul(diff_mu[:, tf.newaxis, :], Gamma_pr_inv_batch), axis=1) 
    norm_mu = tf.reduce_sum(diff_mu * diff_mu_weighted, axis=-1)

    # theta^2 * tr(Gamma_pr^-1 * Gamma)
    prior_var_2 = (theta**2) * tf.reduce_sum(Gamma_pr_inv_batch * Gamma, axis=[-1, -2])
      
    # forward pass
    x_obs_exp = tf.tile(x_obs[tf.newaxis, :, :], [B, 1, 1])
    def eval_pinn(u_in):
        u_exp = tf.tile(u_in[:, tf.newaxis, :], [1, D_y, 1])
        pinn_in = tf.concat([x_obs_exp, u_exp], axis=-1)
        pinn_in_flat = tf.reshape(pinn_in, [-1, 4])
        y_pred_flat = pinn_model(pinn_in_flat)
        return tf.reshape(y_pred_flat, [B, D_y]) 

    # || y - mu_E - F(mu) ||^2_{Gamma_E^-1}
    y_pred_mu = eval_pinn(mu)
    diff_mu_y = y_obs - mu_E - y_pred_mu
    diff_mu_y_weighted = tf.squeeze(tf.matmul(diff_mu_y[:, tf.newaxis, :], Gamma_E_inv_batch), axis=1)
    norm_y_mu = tf.reduce_sum(diff_mu_y * diff_mu_y_weighted, axis=-1)

    # tr(Gamma_E^-1 * J_F * Gamma * J_F^T)
    trace_J_C = tf.constant(0.0, dtype=y_obs.dtype)
    for j in range(D_z):
        C_j = C[:, :, j]
        u_pert = mu + theta * C_j
        y_pred_pert = eval_pinn(u_pert)
        dy_j = y_pred_pert - y_pred_mu
        dy_j_weighted = tf.squeeze(tf.matmul(dy_j[:, tf.newaxis, :], Gamma_E_inv_batch), axis=1)
        trace_J_C += tf.reduce_sum(dy_j * dy_j_weighted, axis=-1)

    total_loss = prior_var_1 + norm_mu + prior_var_2 + norm_y_mu + trace_J_C
    terms = {
        'prior_trace_inv': tf.reduce_mean(prior_var_1),   # θ^2 tr(Γ^-1 Γ_pr)
        'prior_mean':      tf.reduce_mean(norm_mu),       # ||μ − μ_pr||^2_{Γ_pr^-1}
        'prior_trace':     tf.reduce_mean(prior_var_2),   # θ^2 tr(Γ_pr^-1 Γ)
        'data_mean':       tf.reduce_mean(norm_y_mu),     # ||y − μ_E − F(μ)||^2_{Γ_E^-1}
        'data_trace':      tf.reduce_mean(trace_J_C),     # Σ_j ||F(μ+θC_j) − F(μ)||^2_{Γ_E^-1}
    }
    return tf.reduce_mean(total_loss), terms


@tf.function(jit_compile=True)
def train_step_encoder_xla(encoder_model, pinn_model, optimizer, y_obs, x_obs, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, Gamma_E_inv, theta, dtype):
    y_obs = tf.cast(y_obs, dtype)
    x_obs = tf.cast(x_obs, dtype)
    theta = tf.cast(theta, dtype)
    mu_pr = tf.cast(mu_pr, dtype)
    Gamma_pr = tf.cast(Gamma_pr, dtype)
    Gamma_pr_inv = tf.cast(Gamma_pr_inv, dtype)
    mu_E = tf.cast(mu_E, dtype)
    Gamma_E_inv = tf.cast(Gamma_E_inv, dtype)
    with tf.GradientTape() as tape:
        qz = encoder_model(y_obs)
        loss, terms = EncoderLoss(
            qz, x_obs, y_obs, pinn_model, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, Gamma_E_inv, theta
        )

    grads = tape.gradient(loss, encoder_model.trainable_variables)
    grads = [tf.clip_by_norm(g, 5.0) if g is not None else g for g in grads]
    optimizer.apply_gradients(zip(grads, encoder_model.trainable_variables))

    return loss, terms

@tf.function(jit_compile=True)
def val_step_encoder_xla(encoder_model, pinn_model, y_obs, x_obs, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, Gamma_E_inv, theta, dtype):
    y_obs = tf.cast(y_obs, dtype)
    x_obs = tf.cast(x_obs, dtype)
    theta = tf.cast(theta, dtype)
    mu_pr = tf.cast(mu_pr, dtype)
    Gamma_pr = tf.cast(Gamma_pr, dtype)
    Gamma_pr_inv = tf.cast(Gamma_pr_inv, dtype)
    mu_E = tf.cast(mu_E, dtype)
    Gamma_E_inv = tf.cast(Gamma_E_inv, dtype)
    qz = encoder_model(y_obs)
    loss, terms = EncoderLoss(
        qz, x_obs, y_obs, pinn_model, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, Gamma_E_inv, theta
    )
    return loss, terms