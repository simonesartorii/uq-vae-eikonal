import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def make_neg_log_post(decoder_pinn, X_obs, mu_pr, Gamma_pr_inv, mu_E, Gamma_E_inv):
    """
    Returns neg_log_post(u, y) batched:
        u: (B, 2) parameters, y: (B, S) observations (scaled units)
        -> (B,) negative log posterior (up to a constant), in u-space.
    Likelihood: y ~ N(psi(u) + mu_E, Gamma_E).
    """
    S = tf.shape(X_obs)[0]

    def neg_log_post(u, y):
        B = tf.shape(u)[0]
        x_exp = tf.tile(X_obs[tf.newaxis, :, :], [B, 1, 1])
        u_exp = tf.tile(u[:, tf.newaxis, :], [1, S, 1])
        inp = tf.reshape(tf.concat([x_exp, u_exp], axis=-1), [-1, 4])
        y_pred = tf.reshape(decoder_pinn(inp), [B, S])

        r = y - y_pred - mu_E                                          # (B, S)
        nll = 0.5 * tf.reduce_sum(tf.matmul(r, Gamma_E_inv) * r, axis=1)
        d = u - mu_pr
        nlp = 0.5 * tf.reduce_sum(tf.matmul(d, Gamma_pr_inv) * d, axis=1)
        return nll + nlp

    return neg_log_post


def find_map(neg_log_post, y, dtype, n_grid, steps, lr, chunk):
    """
    MAP in u-space computed only from the posterior (likelihood + prior),
    starting from a grid of points independent of the encoder.
    """
    g = np.linspace(0.2, 0.8, n_grid)
    starts = np.array([[a, b] for a in g for b in g], dtype=np.float32)  # (K, 2)
    K = len(starts)

    @tf.function
    def loss_and_grad(z, y_rep):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(neg_log_post(tf.sigmoid(z), y_rep))
        return loss, tape.gradient(loss, z)

    out = []
    for s in range(0, y.shape[0], chunk):
        y_c = y[s:s + chunk]
        B = y_c.shape[0]
        y_rep = tf.repeat(y_c, K, axis=0) # (B*K, S)
        u0 = np.tile(starts, [B, 1]) # (B*K, 2)
        z = tf.Variable(np.log(u0 / (1.0 - u0)), dtype=dtype)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        for _ in range(steps):
            _, grad = loss_and_grad(z, y_rep)
            opt.apply_gradients([(grad, z)])

        u_all = tf.sigmoid(z)
        nlp = neg_log_post(u_all, y_rep).numpy().reshape(B, K)
        u_all = u_all.numpy().reshape(B, K, 2)
        out.append(u_all[np.arange(B), np.argmin(nlp, axis=1)])

    return np.concatenate(out, axis=0) # (N, 2)

def make_mcmc_and_hessian_fn(neg_log_post, dtype, mcmc_steps=300, 
                             mcmc_burnin=100, mcmc_step_size=0.04, eps_cond=1e10):
    """
    Returns a compiled tf.function that runs MCMC and computes the Laplace approximation (Hessian of the negative log-posterior) at the MAP estimate.
    
    Args:
        neg_log_post: function computing the negative log-posterior in u-space
        mcmc_steps: number of MCMC sampling steps
        mcmc_burnin: number of MCMC burn-in steps
        mcmc_step_size: MCMC Random Walk scale parameter
        eps_cond: maximum allowed condition number for the Hessian
    """

    def log_post_u(u_flat, y_i):
        return -neg_log_post(tf.reshape(u_flat, [1, 2]), y_i[tf.newaxis, :])[0]

    def log_post_z(z_flat, y_i):
        u_flat = tf.math.sigmoid(z_flat)
        log_jac = tf.reduce_sum(tf.math.log(u_flat) + tf.math.log(1.0 - u_flat))
        return log_post_u(u_flat, y_i) + log_jac

    @tf.function
    def mcmc_and_hessian(u_map, y_i):
        z_start = tf.math.log(u_map / (1.0 - u_map))

        def target_log_prob(z): 
            return log_post_z(z, y_i)

        kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=mcmc_step_size)
        )
        _, is_accepted = tfp.mcmc.sample_chain(
            num_results=mcmc_steps,
            num_burnin_steps=mcmc_burnin,
            current_state=z_start,
            kernel=kernel,
            trace_fn=lambda _, pkr: pkr.is_accepted
        )
        acc_rate = tf.reduce_mean(tf.cast(is_accepted, dtype))

        # Laplace: Hessian of -log posterior (u-space) at the MAP
        with tf.GradientTape() as t2:
            t2.watch(u_map)
            with tf.GradientTape() as t1:
                t1.watch(u_map)
                loss_val = -log_post_u(u_map, y_i)
            g = t1.gradient(loss_val, u_map)
        hessian = t2.jacobian(g, u_map)
        hessian = 0.5 * (hessian + tf.transpose(hessian))

        hessian_64 = tf.cast(hessian, tf.float64)
        g_64 = tf.cast(g, tf.float64)
        
        newton_step_64 = tf.norm(tf.linalg.solve(hessian_64 + 1e-12 * tf.eye(2, dtype=tf.float64), g_64[:, tf.newaxis]))
        newton_step = tf.cast(newton_step_64, dtype)

        eig = tf.linalg.eigvalsh(hessian_64)
        min_eig, max_eig = tf.cast(eig[0], dtype), tf.cast(eig[-1], dtype)
        cond = max_eig / (min_eig + 1e-12)
        
        hess_ok = tf.logical_and(tf.logical_and(tf.math.is_finite(cond), min_eig > 0.0), cond < eps_cond)
        
        cov_laplace_64 = tf.cond(hess_ok, 
                                 lambda: tf.linalg.inv(hessian_64), 
                                 lambda: tf.zeros((2, 2), dtype=tf.float64))
        cov_laplace = tf.cast(cov_laplace_64, dtype)

        return hess_ok, cov_laplace, acc_rate, newton_step

    return mcmc_and_hessian

def make_forward(decoder_pinn, X_obs):
    S = tf.shape(X_obs)[0]
    def forward(u): # (B,2) -> (B,S)
        B = tf.shape(u)[0]
        x_exp = tf.tile(X_obs[tf.newaxis], [B, 1, 1])
        u_exp = tf.tile(u[:, tf.newaxis, :], [1, S, 1])
        inp = tf.reshape(tf.concat([x_exp, u_exp], -1), [-1, 4])
        return tf.reshape(decoder_pinn(inp), [B, S])
    return forward


def make_gn_cov_fn(forward, Gamma_E_inv, Gamma_pr_inv):
    @tf.function
    def gn_cov(u_map): # (2,)
        u = u_map[tf.newaxis, :]
        with tf.GradientTape() as tape:
            tape.watch(u)
            psi = forward(u) # (1,S)
        J = tf.squeeze(tape.jacobian(psi, u), axis=[0, 2])   # (S,2)
        H = tf.transpose(J) @ Gamma_E_inv @ J + Gamma_pr_inv
        H = 0.5 * (H + tf.transpose(H))
        return tf.linalg.inv(H)
    return gn_cov