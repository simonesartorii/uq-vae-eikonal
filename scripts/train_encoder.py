import os
import sys
import random
import time
import numpy as np
import tensorflow as tf

from src.networks.Encoder import Encoder
from src.networks.DecoderPINN import PINN
from src.optimization.EncoderTrainer import EncoderTrainer
from src.data.encoder_dataloader import load_encoder_data
from src.visualization.historyplots import plot_encoder_adam_history, plot_encoder_terms
from src.hyperparameters import hypers_uqvae as h
from src.visualization.pdeplots import N_GRID

def train_and_evaluate_encoder(decoder_model, output_dir: str, result_dir: str, data_dir):
    tf.keras.backend.clear_session()

    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    tf.keras.backend.set_floatx('float64')
    number_type = tf.float64

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    test_path = os.path.join(data_dir, 'test.npz')
    scale_path = os.path.join(data_dir, 'u_scale.npy')
    base_data_dir = os.path.dirname(data_dir)
    sensor_idx_path = os.path.join(base_data_dir, 'sensor_idx.npy')

    (y_train_clean, u_train), (y_obs_val, u_val), (y_obs_test, u_test), X_obs, sensor_idx, U_SCALE = load_encoder_data(
        train_path=train_path, val_path=val_path, test_path=test_path,
        scale_path=scale_path, n_sensors=h.N_SENSORS,
        region_bounds=h.SENSOR_REGION, sensor_idx_path=sensor_idx_path,
        noise_level=h.NOISE, dtype=number_type
    )

    noise_global_path = os.path.join(os.path.dirname(data_dir), 'noise.npz')
    sigma_noise = float(np.load(noise_global_path)['noise_std_physical']) / float(U_SCALE)
    print(f"sigma_noise (scaled): {sigma_noise:.5f}")


    decoder_pinn = PINN(h.N_NEURONS_DECODER, U_SCALE)
    decoder_pinn(tf.zeros((1, 4), dtype=number_type))
    decoder_weights_path = os.path.join(output_dir, 'decoder_lbfgs.weights.h5')
    if not os.path.exists(decoder_weights_path):
        decoder_weights_path = os.path.join(output_dir, 'decoder_adam.weights.h5')
    decoder_pinn.load_weights(decoder_weights_path)
    print(f"Decoder weights loaded: {decoder_weights_path}")

    decoder_pinn.trainable = False
    
    encoder = Encoder(n_neurons=h.N_NEURONS_ENCODER, latent_dim=h.D_z)
    encoder(tf.zeros((1, h.N_SENSORS), dtype=number_type))
    encoder.init_output_layer(bias=h.BIAS, weight_scale=h.WEIGHT_SCALE_ENCODER_OUT)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        h.LR_ENCODER, decay_steps=h.LR_DECAY_STEPS_ENCODER, decay_rate=h.LR_DECAY_RATE_ENCODER, staircase=False
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-12)
    optimizer.build(encoder.trainable_variables)

    mu_pr = tf.constant(h.MU_PR, dtype=number_type)
    sigma_pr = tf.constant(h.SIGMA_PR, dtype=number_type)
    Gamma_pr = tf.eye(h.D_z, dtype=number_type) * (sigma_pr ** 2)
    Gamma_pr_inv = tf.linalg.inv(Gamma_pr)

    B_train = tf.shape(u_train)[0]
    D_y = tf.shape(X_obs)[0]
    
    x_obs_exp = tf.tile(X_obs[tf.newaxis, :, :], [B_train, 1, 1])
    u_exp = tf.tile(u_train[:, tf.newaxis, :], [1, D_y, 1])
    pinn_in = tf.concat([x_obs_exp, u_exp], axis=-1)
    pinn_in_flat = tf.reshape(pinn_in, [-1, 4])
    
    y_pred_flat = decoder_pinn(pinn_in_flat)
    y_pred = tf.reshape(y_pred_flat, [B_train, D_y])
    
    residuals_clean = y_train_clean - y_pred
    mu_dec = tf.reduce_mean(residuals_clean, axis=0)

    residuals_centered = residuals_clean - mu_dec
    Gamma_decoder = tf.matmul(residuals_centered, residuals_centered, transpose_a=True) / tf.cast(B_train - 1, number_type)

    sigma_noise_tf = tf.constant(sigma_noise, dtype=number_type)
    Gamma_noise = tf.square(sigma_noise_tf) * tf.eye(D_y, dtype=number_type)

    Gamma_E = Gamma_decoder + Gamma_noise
    Gamma_E_inv = tf.linalg.inv(Gamma_E)

    mu_E = mu_dec

    gamma_save_path = os.path.join(output_dir, 'gamma_E.npz')
    np.savez(gamma_save_path, mu_E=mu_E.numpy(), Gamma_E_inv=Gamma_E_inv.numpy())
    
    @tf.function
    def augment(y, u):
        if h.NOISE > 0.0:
            noise = tf.random.normal(shape=tf.shape(y), mean=0.0, stddev=sigma_noise, dtype=number_type)
            y_noisy = y + noise
        else:
            y_noisy = y
        return y_noisy, u
    
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((y_train_clean, u_train))
        .shuffle(buffer_size=10000, seed=42)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(h.BATCH_SIZE_ENCODER, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))

    t0_enc = time.time()
    trainer = EncoderTrainer(encoder=encoder, decoder=decoder_pinn, output_dir=output_dir, result_dir=result_dir)
    trainer.train_adam(optimizer=optimizer, epochs=h.EPOCHS_ENCODER_ADAM,
                       train_dataset=train_dataset, 
                       y_obs_val=y_obs_val, 
                       x_obs=X_obs,
                       mu_pr=mu_pr, Gamma_pr=Gamma_pr, Gamma_pr_inv=Gamma_pr_inv,
                       mu_E=mu_E, 
                       Gamma_E_inv=Gamma_E_inv, 
                       theta=h.THETA_ENCODER,
                       patience=h.PATIENCE_ENCODER_ADAM, 
                       save_freq=h.SAVE_FREQ_ENCODER_ADAM,
                       dtype=number_type)
    t_enc = time.time() - t0_enc
    
    os.makedirs(result_dir, exist_ok=True)
    history_adam_path = os.path.join(result_dir, 'encoder_adam.loss.npz')
    np.savez(history_adam_path, **trainer.history_adam)
    plot_encoder_adam_history(trainer.history_adam, save_freq=h.SAVE_FREQ_ENCODER_ADAM)
    plot_encoder_terms(trainer.history_adam, save_freq=h.SAVE_FREQ_ENCODER_ADAM)
    
    qz_val = encoder(y_obs_val)
    u_pred_val = qz_val.mean().numpy()

    mae_u_val = np.mean(np.abs(u_pred_val - u_val.numpy()), axis=0)
    print(f"MAE Validation on cx: {mae_u_val[0]:.4f}")
    print(f"MAE Validation on cy: {mae_u_val[1]:.4f}\n")

    print("\nTrue vs Pred (first 5 samples Validation):")
    for i in range(min(5, len(u_val))):
        print(f"True [cx, cy]: {u_val[i].numpy()} | Pred [cx, cy]: {u_pred_val[i]}")

    qz_test = encoder(y_obs_test)
    u_pred_test = qz_test.mean().numpy()
 
    mae_u_test = np.mean(np.abs(u_pred_test - u_test.numpy()), axis=0)
    mae_cx_test = float(mae_u_test[0])
    mae_cy_test = float(mae_u_test[1])    
    print(f"\nMAE Test on cx: {mae_cx_test:.4f}")
    print(f"MAE Test on cy: {mae_cy_test:.4f}")

    mu = tf.cast(qz_test.mean(), number_type)            
    cov = tf.cast(qz_test.covariance(), number_type)    
    diff = u_test - mu    
    inv_cov = tf.linalg.inv(cov) 

    mahalanobis_sq = tf.einsum('ni,nij,nj->n', diff, inv_cov, diff)
    coverage_2d_68 = tf.reduce_mean(tf.cast(mahalanobis_sq <= 2.296, tf.float32)) * 100.0
    coverage_2d_95 = tf.reduce_mean(tf.cast(mahalanobis_sq <= 5.991, tf.float32)) * 100.0
    coverage_2d_68_val = float(coverage_2d_68.numpy())
    coverage_2d_95_val = float(coverage_2d_95.numpy())
    print(f"Coverage (target 68.3%): {coverage_2d_68_val:.2f}%")
    print(f"Coverage (target 95.0%): {coverage_2d_95_val:.2f}%")
    
    return encoder, mae_cx_test, mae_cy_test, coverage_2d_68_val, coverage_2d_95_val, t_enc