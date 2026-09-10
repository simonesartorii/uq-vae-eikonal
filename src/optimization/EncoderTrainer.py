import time
import tensorflow as tf
import os

from src.optimization.Trainer import Trainer
from src.optimization.EncoderTrainingFunctions import train_step_encoder_xla, val_step_encoder_xla

class EncoderTrainer(Trainer):
    def __init__(self,
                 encoder: tf.keras.Model,
                 decoder: tf.keras.Model,
                 pipeline_id: str):

        super().__init__(pipeline_id)

        self.encoder = encoder
        self.decoder = decoder

    def train_adam(self,
                   optimizer: tf.keras.optimizers.Optimizer,
                   epochs: int,
                   train_dataset: tf.data.Dataset,
                   y_val: tf.Tensor,
                   x_obs: tf.Tensor,
                   mu_pr: tf.Tensor,
                   Gamma_pr: tf.Tensor,
                   Gamma_pr_inv: tf.Tensor,
                   mu_E: tf.Tensor,
                   sigma_noise: float,
                   u_scale: float,
                   patience: int = 400,
                   save_freq: int = 10):

        print('\nStarting Encoder Adam optimization...')
        self.history_adam = {'total': [], 'data_fid': [], 'val': [], 'val_dfid': []}

        t0 = time.time()
        best_val_loss = float('inf')
        counter = 0

        save_path = self._weights_path('encoder_adam')

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_dfid_avg = tf.keras.metrics.Mean()

        for epoch in range(epochs + 1):

            epoch_loss_avg.reset_state()
            epoch_dfid_avg.reset_state()

            for y_batch, u_batch in train_dataset:
                # training step
                loss, t1, data_fid, n_mu, t2 = train_step_encoder_xla(
                    self.encoder, self.decoder, optimizer, y_batch, x_obs,
                    mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise, u_scale
                )

                epoch_loss_avg.update_state(loss)
                epoch_dfid_avg.update_state(data_fid)

            # val step
            val_loss, val_dfid = val_step_encoder_xla(
                self.encoder, self.decoder, y_val, x_obs,
                mu_pr, Gamma_pr, Gamma_pr_inv, mu_E, sigma_noise, u_scale
            )
            val_loss_val = val_loss.numpy()

            # early stopping
            best_val_loss, counter, improved = self._check_early_stopping(
                val_loss_val, best_val_loss, counter, patience
            )

            if improved:
                self.encoder.save_weights(save_path)
            elif counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4e}")
                break

            # saving history
            if epoch % save_freq == 0:
                self.history_adam['total'].append(float(epoch_loss_avg.result().numpy()))
                self.history_adam['data_fid'].append(float(epoch_dfid_avg.result().numpy()))
                self.history_adam['val'].append(float(val_loss_val))
                self.history_adam['val_dfid'].append(float(val_dfid.numpy()))

            if epoch % 100 == 0:
                print(f'Epoch {epoch:4d} | Loss: {epoch_loss_avg.result().numpy():.4e} | Data Fid: {epoch_dfid_avg.result().numpy():.4e} | Val Loss: {val_loss_val:.4e} | Val Data Fid: {val_dfid.numpy():.4e}')

        if os.path.exists(save_path):
            self.encoder.load_weights(save_path)
            print(f"Restored best encoder weights from early stopping (val_loss: {best_val_loss:.4e})")
        print(f"Encoder Adam optimization finished in {(time.time() - t0)/60:.2f} minutes.")

        w_path, h_path = self.save_weights_and_history(model=self.encoder, history_dict=self.history_adam, tag='encoder_adam')

        data = {
            "optimizer": "Adam",
            "epochs_run": epoch,
            "patience": patience,
            "sigma_noise": float(sigma_noise),
            "u_scale": float(u_scale),
            "execution_time_minutes": round((time.time() - t0) / 60, 2),
            "weights_file": w_path,
            "history_file": h_path,
            "metrics": {
                "best_val_loss": float(best_val_loss),
                "final_train_loss": float(epoch_loss_avg.result().numpy()),
                "final_data_fid": float(epoch_dfid_avg.result().numpy()),
                "final_val_loss": float(val_loss_val),
                "final_val_dfid": float(val_dfid.numpy())
            }
        }
        
        self.update_experiment_log(phase_name="encoder_vae_adam", phase_data=data)