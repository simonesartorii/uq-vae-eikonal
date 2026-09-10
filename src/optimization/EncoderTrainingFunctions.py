import tensorflow as tf

def EncoderLoss(qz, y_true, pinn_model, x_obs, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise, u_scale, theta=1e-4):
    
    B = tf.shape(y_true)[0]
    D_y = tf.shape(y_true)[1]
    D_z = 2
    
    mu = qz.mean() # (batch_size, D_z)
    Gamma = qz.covariance() # (batch_size, D_z, D_z)
    C = qz.scale.to_dense() # (batch_size, D_z, D_z) Cholesky factor

    
    # prior covariance and precision bcast
    Gamma_pr_batch = tf.tile(Gamma_pr[tf.newaxis, :, :], [B, 1, 1])
    Gamma_pr_inv_batch = tf.tile(Gamma_pr_inv[tf.newaxis, :, :], [B, 1, 1])

    # 1. tr(Gamma^-1 * Gamma_pr)
    X1 = tf.linalg.cholesky_solve(C, Gamma_pr_batch) 
    prior_var_1 = (theta**2) * tf.linalg.trace(X1)

    # 2. || mu - mu_pr ||^2_{Gamma_pr^-1}
    diff_mu = mu - mu_pr 
    diff_mu_weighted = tf.squeeze(tf.matmul(diff_mu[:, tf.newaxis, :], Gamma_pr_inv_batch), axis=1) 
    norm_mu = tf.reduce_sum(diff_mu * diff_mu_weighted, axis=-1)

    # 3. theta^2 * tr(Gamma_pr^-1 * Gamma)
    prior_var_2 = (theta**2) * tf.reduce_sum(Gamma_pr_inv_batch * Gamma, axis=[-1, -2])
      
    # forward 
    x_obs_exp = tf.tile(x_obs[tf.newaxis, :, :], [B, 1, 1])
    def eval_pinn(u_in):
        u_exp = tf.tile(u_in[:, tf.newaxis, :], [1, D_y, 1])
        pinn_in = tf.concat([x_obs_exp, u_exp], axis=-1)
        pinn_in_flat = tf.reshape(pinn_in, [-1, 4])
        y_pred_flat = pinn_model(pinn_in_flat)
        return tf.reshape(y_pred_flat, [B, D_y]) 

    # 4. || y - mu_E - F(mu) ||^2_{Gamma_E^-1}
    y_pred_mu = eval_pinn(mu)
    diff_mu_y = y_true - mu_E - y_pred_mu
    norm_y_mu = tf.reduce_sum(tf.square(diff_mu_y), axis=-1) / tf.square(sigma_noise)

   
    trace_J_C = 0.0
    for j in range(D_z):
        C_j = C[:, :, j]
        u_pert = mu + theta * C_j
        y_pred_pert = eval_pinn(u_pert)
        dy_j = y_pred_pert - y_pred_mu
        trace_J_C += tf.reduce_sum(tf.square(dy_j), axis=-1) / tf.square(sigma_noise)

    total_loss = prior_var_1 + norm_mu + prior_var_2 + norm_y_mu + trace_J_C

    return tf.reduce_mean(total_loss), tf.reduce_mean(prior_var_1), tf.reduce_mean(norm_y_mu + trace_J_C), tf.reduce_mean(norm_mu), tf.reduce_mean(prior_var_2)


@tf.function(jit_compile=True)
def train_step_encoder_xla(encoder_model, pinn_model, optimizer, y_true, x_obs, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise, u_scale):
    with tf.GradientTape() as tape:
        qz = encoder_model(y_true)
        loss, t1, data_fid, n_mu, t2 = EncoderLoss(
            qz, y_true, pinn_model, x_obs, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise, u_scale
        )

    grads = tape.gradient(loss, encoder_model.trainable_variables)
    grads = [tf.clip_by_norm(g, 5.0) if g is not None else g for g in grads]
    optimizer.apply_gradients(zip(grads, encoder_model.trainable_variables))

    return loss, t1, data_fid, n_mu, t2

@tf.function(jit_compile=True)
def val_step_encoder_xla(encoder_model, pinn_model, y_true, x_obs, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise, u_scale):
    qz = encoder_model(y_true)
    loss, t1, data_fid, n_mu, t2 = EncoderLoss(
        qz, y_true, pinn_model, x_obs, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise, u_scale
    )
    return loss, data_fid


# vecchie loss con monte carlo / errori, andrebbero sistemate
# def EncoderLoss(qz, y_true, y_pred_k, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise):
#     """
#     Computes the loss function for the encoder in the eUQ-VAE framework
#     """
#     B = tf.shape(y_true)[0]
#     mu = qz.mean() # (batch_size, D_z)
#     Gamma = qz.covariance() # (batch_size, D_z, D_z)
#     C = qz.scale.to_dense() # (batch_size, D_z, D_z) Cholesky factor

#     # prior covariance and precision bcast
#     Gamma_pr_batch = tf.tile(Gamma_pr[tf.newaxis, :, :], [B, 1, 1])
#     Gamma_pr_inv_batch = tf.tile(Gamma_pr_inv[tf.newaxis, :, :], [B, 1, 1])

#     # 1. tr(Gamma^-1 * Gamma_pr)
#     X1 = tf.linalg.cholesky_solve(C, Gamma_pr_batch) 
#     trace1 = tf.linalg.trace(X1) 

#     # 2. || y - mu_E - psi(u^k) ||^2_{Gamma_E^-1}
#     diff_y = y_true[:, tf.newaxis, :] - mu_E - y_pred_k 
    
#     norm_y = tf.reduce_sum(tf.square(diff_y), axis=-1) / tf.square(sigma_noise)
#     data_fidelity = tf.reduce_mean(norm_y, axis=1) 

#     # 3. || mu - mu_pr ||^2_{Gamma_pr^-1} 
#     diff_mu = mu - mu_pr 
#     diff_mu_weighted = tf.squeeze(tf.matmul(diff_mu[:, tf.newaxis, :], Gamma_pr_inv_batch), axis=1) 
#     norm_mu = tf.reduce_sum(diff_mu * diff_mu_weighted, axis=-1) 

#     # 4. tr(Gamma_pr^-1 * Gamma) 
#     X2 = tf.matmul(Gamma_pr_inv_batch, Gamma) 
#     trace2 = tf.linalg.trace(X2) 

#     total_loss = ((1.0 - alpha) * trace1 
#                 + alpha * data_fidelity 
#                 + alpha * (norm_mu + trace2))
    
#     return total_loss, trace1, data_fidelity, norm_mu, trace2


# @tf.function(jit_compile=True)
# def train_step_encoder_xla(encoder_model, pinn_model, optimizer, y_true, K, x_obs, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise):
#     B = tf.shape(y_true)[0]
#     D_y = tf.shape(y_true)[1]
    
#     with tf.GradientTape() as tape:
#         qz = encoder_model(y_true)
#         u_samples = qz.sample(K)
#         u_samples = tf.transpose(u_samples, perm=[1, 0, 2]) 
        
#         x_obs_exp = tf.tile(x_obs[tf.newaxis, tf.newaxis, :, :], [B, K, 1, 1])
#         u_exp = tf.tile(u_samples[:, :, tf.newaxis, :], [1, 1, D_y, 1])
        
#         pinn_in = tf.concat([x_obs_exp, u_exp], axis=-1)
#         pinn_in_flat = tf.reshape(pinn_in, [-1, 4])

#         y_pred_flat = pinn_model(pinn_in_flat)
#         y_pred_k = tf.reshape(y_pred_flat, [B, K, D_y])
        
#         loss, t1, data_fid, n_mu, t2 = EncoderLoss(qz, y_true, y_pred_k, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise)
#         mean_loss = tf.reduce_mean(loss)
        
#     grads = tape.gradient(mean_loss, encoder_model.trainable_variables)
#     grads = [tf.clip_by_norm(g, 5.0) if g is not None else g for g in grads]
#     optimizer.apply_gradients(zip(grads, encoder_model.trainable_variables))

#     return mean_loss, tf.reduce_mean(t1), tf.reduce_mean(data_fid), tf.reduce_mean(n_mu), tf.reduce_mean(t2)


# @tf.function(jit_compile=True)
# def val_step_encoder_xla(encoder_model, pinn_model, y_true, K, x_obs, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise):
#     B = tf.shape(y_true)[0]
#     D_y = tf.shape(y_true)[1]
    
#     qz = encoder_model(y_true)
#     u_samples = qz.sample(K)
#     u_samples = tf.transpose(u_samples, perm=[1, 0, 2]) 
    
#     x_obs_exp = tf.tile(x_obs[tf.newaxis, tf.newaxis, :, :], [B, K, 1, 1])
#     u_exp = tf.tile(u_samples[:, :, tf.newaxis, :], [1, 1, D_y, 1])
    
#     pinn_in = tf.concat([x_obs_exp, u_exp], axis=-1)
#     pinn_in_flat = tf.reshape(pinn_in, [-1, 4])
    
#     y_pred_flat = pinn_model(pinn_in_flat)
#     y_pred_k = tf.reshape(y_pred_flat, [B, K, D_y])
    
#     loss, t1, data_fid, n_mu, t2 = EncoderLoss(qz, y_true, y_pred_k, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise)
#     return tf.reduce_mean(loss), tf.reduce_mean(data_fid)


# # import tensorflow as tf

# # def EncoderLoss(qz, y_true, u_true, y_pred_k, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise):
# #     """
# #     Computes the loss function for the encoder in the eUQ-VAE framework
    
# #     Args:
# #         qz: tfd.MultivariateNormalTriL q_phi(u|y) (mu and C))
# #         y_true: (batch_size, D_y) tensor containing the observed data y
# #         u_true: (batch_size, D_u) tensor containing the true values of u
# #         y_pred_k: (batch_size, K, D_y) tensor containing the decoder evaluations psi(u^k) for K samples
# #         alpha
# #         mu_pr: (batch_size, D_z) tensor containing the prior mean mu_pr
# #         Gamma_pr: (batch_size, D_z, D_z) tensor containing the prior covariance Gamma_pr
# #         Gamma_pr_inv: (batch_size, D_z, D_z) tensor containing the prior precision Gamma_pr^-1
# #         mu_E: float or tensor, the mean of the noise epsilon
# #         L_prec: (D_y, D_y) tensor, Cholesky factor of the precision matrix tf.linalg.cholesky(Gamma_E_inv)
# #     """

# #     mu = qz.mean() # (batch_size, D_z)
# #     Gamma = qz.covariance() # (batch_size, D_z, D_z)
# #     C = qz.scale.to_dense() # (batch_size, D_z, D_z) Cholesky factor

# #     # 1. tr(Gamma^-1 * Gamma_pr)
# #     X1 = tf.linalg.cholesky_solve(C, Gamma_pr) # (batch_size, D_z, D_z)
# #     trace1 = tf.linalg.trace(X1) # (batch_size,)

# #     # 2. || y - mu_E - psi(u^k) ||^2_{Gamma_E^-1}
# #     # (batch_size, 1, D_y) - (batch_size, K, D_y)
# #     diff_y = y_true[:, tf.newaxis, :] - mu_E - y_pred_k # (batch_size, K, D_y)
    
# #     diff_y_transformed = tf.matmul(diff_y, L_prec)# (batch_size, K, D_y)
    
# #     norm_y = tf.reduce_sum(diff_y_transformed ** 2, axis=-1) # (batch_size, K)
# #     data_fidelity = tf.reduce_mean(norm_y, axis=1) # (batch_size,)

# #     # 3. || mu - mu_pr ||^2_{Gamma_pr^-1} 
# #     diff_mu = mu - mu_pr  # (batch_size, D_z)
    
# #     # (batch_size, 1, D_z) x (batch_size, D_z, D_z)
# #     diff_mu_weighted = tf.squeeze(tf.matmul(diff_mu[:, tf.newaxis, :], Gamma_pr_inv), axis=1) # (batch_size, D_z)
    
# #     norm_mu = tf.reduce_sum(diff_mu * diff_mu_weighted, axis=-1) # (batch_size,)

# #     # 4. tr(Gamma_pr^-1 * Gamma) 
# #     X2 = tf.matmul(Gamma_pr_inv, Gamma) # (batch_size, D_z, D_z)
# #     trace2 = tf.linalg.trace(X2) # (batch_size,)

# #     total_loss = (1-alpha) * trace1 + alpha * data_fidelity + alpha * (norm_mu + trace2) # (batch_size,)
    
# #     return total_loss


# # # TRAIN STEP
# # def train_step_encoder(encoder_model, pinn_model, optimizer, y_true, K, x_obs, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, L_prec):
    
# #     with tf.GradientTape() as tape:
# #         tape.watch(encoder_model.trainable_variables)
        
# #         loss, t1, data_fid, n_mu, t2 = EncoderLoss(encoder_model, y_true, u_true, K, x_obs, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, L_prec)
# #         mean_loss = tf.reduce_mean(loss)
        
# #     grads = tape.gradient(mean_loss, encoder_model.trainable_variables)
# #     optimizer.apply_gradients(zip(grads, encoder_model.trainable_variables))

# #     return mean_loss, tf.reduce_mean(t1), tf.reduce_mean(data_fid), tf.reduce_mean(n_mu), tf.reduce_mean(t2)


# # # VALIDATION STEP
# # def val_step_encoder(encoder_model, pinn_model, y_true, u_true, K, x_obs, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, L_prec):
# #     loss, t1, data_fid, n_mu, t2 = EncoderLoss(encoder_model, y_true, u_true, K, x_obs, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, L_prec)
# #     return tf.reduce_mean(loss)


# # # XLA TRAINING STEP
# # @tf.function(jit_compile=True)
# # def train_step_encoder_xla(encoder_model, pinn_model, optimizer, y_true, u_true, K, x_obs, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, L_prec):
# #     """
# #     XLA-compiled version
# #     """
# #     with tf.GradientTape() as tape:
# #         loss, t1, data_fid, n_mu, t2 = EncoderLoss(encoder_model, y_true, u_true, K, x_obs, alpha, mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, L_prec)
# #         mean_loss = tf.reduce_mean(loss)
        
# #     grads = tape.gradient(mean_loss, encoder_model.trainable_variables)
# #     optimizer.apply_gradients(zip(grads, encoder_model.trainable_variables))

# #     return mean_loss, tf.reduce_mean(t1), tf.reduce_mean(data_fid), tf.reduce_mean(n_mu), tf.reduce_mean(t2)