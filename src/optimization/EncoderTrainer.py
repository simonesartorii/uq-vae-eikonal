import time
import tensorflow as tf
import os

from src.optimization.Trainer import Trainer
from src.optimization.EncoderTrainingFunctions import train_step_encoder_xla, val_step_encoder_xla

class EncoderTrainer(Trainer):
    def __init__(self, encoder: tf.keras.Model, 
                 decoder: tf.keras.Model, 
                 output_dir: str,
                 result_dir: str):
        
        super().__init__(output_dir, result_dir)
        
        self.encoder = encoder
        self.decoder = decoder

    def train_adam(self,
                   optimizer: tf.keras.optimizers.Optimizer,
                   epochs: int,
                   train_dataset: tf.data.Dataset,
                   y_obs_val: tf.Tensor,
                   x_obs: tf.Tensor,
                   mu_pr: tf.Tensor,
                   Gamma_pr: tf.Tensor,
                   Gamma_pr_inv: tf.Tensor,
                   mu_E: tf.Tensor,
                   Gamma_E_inv: tf.Tensor,
                   theta: float = 1e-4,
                   patience: int = 400,
                   save_freq: int = 10,
                   dtype=tf.float32):

        print('\nStarting Encoder Adam optimization...')
        self.history_adam = {'total': [], 'data_fid': [], 'val': [], 'val_dfid': []}
        TERMS = ['prior_trace_inv', 'prior_mean', 'prior_trace', 'data_mean', 'data_trace']
        for k in TERMS:
            self.history_adam[k] = []
            self.history_adam['val_' + k] = []

        term_avg = {k: tf.keras.metrics.Mean() for k in TERMS}

        t0 = time.time()
        best_val_loss = float('inf')
        counter = 0

        save_path = self._weights_path('encoder_adam')

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_dfid_avg = tf.keras.metrics.Mean()
        
        theta_tf = tf.constant(theta, dtype=dtype)

        for epoch in range(epochs + 1):
            for k in TERMS:
                term_avg[k].reset_state()
            epoch_loss_avg.reset_state()
            epoch_dfid_avg.reset_state()

            for y_obs_batch, u_batch in train_dataset:
                y_obs_batch = tf.cast(y_obs_batch, dtype=dtype)
                # Training step
                loss, terms = train_step_encoder_xla(
                    self.encoder, self.decoder, optimizer, y_obs_batch, x_obs,
                    mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, Gamma_E_inv, theta_tf, dtype
                )

                epoch_loss_avg.update_state(loss)
                epoch_dfid_avg.update_state(terms['data_mean'] + terms['data_trace'])
                for k in TERMS:
                    term_avg[k].update_state(terms[k])

            # Validation step
            val_loss, val_terms = val_step_encoder_xla(
                self.encoder, self.decoder, y_obs_val, x_obs,
                mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, Gamma_E_inv, theta_tf, dtype
            )
            val_dfid = val_terms['data_mean'] + val_terms['data_trace']
            val_loss_val = val_loss.numpy()

            # Early stopping
            best_val_loss, counter, improved = self._check_early_stopping(
                val_loss_val, best_val_loss, counter, patience
            )

            if improved:
                self.encoder.save_weights(save_path)
            elif counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4e}")
                break

            # Saving history
            if epoch % save_freq == 0:
                self.history_adam['total'].append(float(epoch_loss_avg.result().numpy()))
                self.history_adam['data_fid'].append(float(epoch_dfid_avg.result().numpy()))
                self.history_adam['val'].append(float(val_loss_val))
                self.history_adam['val_dfid'].append(float(val_dfid.numpy()))
                for k in TERMS:
                    self.history_adam[k].append(float(term_avg[k].result().numpy()))
                    self.history_adam['val_' + k].append(float(val_terms[k].numpy()))

            if epoch % 100 == 0:
                print(f'Epoch {epoch:4d} | Loss: {epoch_loss_avg.result().numpy():.4e} | Val Loss: {val_loss_val:.4e}')
                print('   train: ' + ' | '.join(f'{k}: {term_avg[k].result().numpy():.3e}' for k in TERMS))
                print('   val:   ' + ' | '.join(f'{k}: {val_terms[k].numpy():.3e}' for k in TERMS))

        if os.path.exists(save_path):
            self.encoder.load_weights(save_path)
            print(f"Restored best encoder weights from early stopping (val_loss: {best_val_loss:.4e})")
        print(f"Encoder Adam optimization finished in {(time.time() - t0)/60:.2f} minutes.")

        w_path, h_path = self.save_weights_and_history(model=self.encoder, history_dict=self.history_adam, tag='encoder_adam')
