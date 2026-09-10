import time
import os
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Dict, Callable

from src.optimization.Trainer import Trainer
from src.optimization.DecoderTrainingFunctions import train_step_xla, val_step_xla, loss_fn


class DecoderTrainer(Trainer):
    def __init__(self,
                 model: tf.keras.Model,

                 pipeline_id: str):

        super().__init__(pipeline_id)

        self.model = model

    def train_adam(self,
                   optimizer: tf.keras.optimizers.Optimizer,
                   epochs: int,
                   x_train: tf.Tensor,
                   u_train: tf.Tensor,
                   x_val: tf.Tensor,
                   u_val: tf.Tensor,
                   physics_fn : Callable,
                   lambdas: Dict[str, float], 
                   point_generator_fn: Callable,
                   batch_size: int = 8192,
                   patience: int = 1000,
                   number_type: tf.DType = tf.float32):

        print('\nStarting Adam optimization...')
        t0 = time.time()
        self.history_adam = {'total': [], 'pde': [], 'dir': [], 'neu': [], 'data': [], 'val': []}
        best_val_loss = float('inf')
        counter = 0

        save_path = self._weights_path('decoder_adam')

        # loss weights
        LAMBDA_PDE = tf.constant(lambdas['pde'], dtype=number_type)
        LAMBDA_DIR = tf.constant(lambdas['dir'], dtype=number_type)
        LAMBDA_NEU = tf.constant(lambdas['neu'], dtype=number_type)
        LAMBDA_DATA = tf.constant(lambdas['data'], dtype=number_type)


        for epoch in range(epochs + 1):
            # random batch from training data
            idx = tf.random.uniform(shape=[batch_size], minval=0, maxval=tf.shape(x_train)[0], dtype=tf.int32)
            x_train_batch = tf.gather(x_train, idx)
            u_train_batch = tf.gather(u_train, idx)

            # generate 4d points for this epoch + cast
            x_int, x_boundary, n_boundary_out, num_neumann = point_generator_fn()

            x_int_tf = tf.cast(x_int, dtype=number_type)
            x_bound_tf = tf.cast(x_boundary, dtype=number_type)
            n_bound_tf = tf.cast(n_boundary_out, dtype=number_type)

            # training step
            loss, lpde, ldir, lneu, ldata = train_step_xla(
                self.model, optimizer, x_int_tf, x_bound_tf, n_bound_tf, num_neumann, x_train_batch, u_train_batch,
                physics_fn, LAMBDA_PDE, LAMBDA_DIR, LAMBDA_NEU, LAMBDA_DATA
            )

            # validation step + early stopping
            val_loss = val_step_xla(self.model, x_val, u_val)
            val_loss_val = val_loss.numpy()

            best_val_loss, counter, improved = self._check_early_stopping(val_loss_val, best_val_loss, counter, patience)
            if improved:
                self.model.save_weights(save_path)
            elif counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4e}")
                break

            # record history and logging
            if epoch % 10 == 0:
                self.history_adam['total'].append(loss.numpy())
                self.history_adam['pde'].append(lpde.numpy())
                self.history_adam['dir'].append(ldir.numpy())
                self.history_adam['neu'].append(lneu.numpy())
                self.history_adam['data'].append(ldata.numpy())
                self.history_adam['val'].append(val_loss_val)

            if epoch % 500 == 0:
                print(f'Epoch {epoch:4d} | Loss: {loss.numpy():.4e} | PDE: {lpde.numpy():.4e} | Dir: {ldir.numpy():.4e} | Neu: {lneu.numpy():.4e} | Data: {ldata.numpy():.4e} | Val: {val_loss_val:.4e}')

        if os.path.exists(save_path):
            self.model.load_weights(save_path)
            print(f"Restored best model weights from early stopping (val_loss: {best_val_loss:.4e})")
            
        print(f"Adam optimization finished in {(time.time() - t0)/60:.2f} minutes.")

        # savings 
        w_path, h_path = self.save_weights_and_history(self.model, self.history_adam, 'decoder_adam')

        data = {
            "optimizer": "Adam",
            "epochs_run": epoch,
            "batch_size": batch_size,
            "patience": patience,
            "execution_time_minutes": round((time.time() - t0) / 60, 2),
            "weights_file": w_path,
            "history_file": h_path,
            "lambdas": lambdas,
            "metrics": {
                "best_val_loss": float(best_val_loss),
                "final_train_loss": float(loss.numpy()),
                "final_pde_loss": float(lpde.numpy()),
                "final_dir_loss": float(ldir.numpy()),
                "final_neu_loss": float(lneu.numpy()),
                "final_data_loss": float(ldata.numpy())
            }
        }
        
        self.update_experiment_log(phase_name="decoder_adam", phase_data=data)

    def train_lbfgs(self,
                    x_int,
                    x_boundary,
                    n_boundary,
                    num_neumann,
                    x_data,
                    u_data,
                    x_val,
                    u_val,
                    physics_fn: Callable,
                    lambdas: Dict[str, float],
                    max_iterations: int = 1500,
                    num_correction_pairs: int = 50,
                    tolerance: float = 1e-12,
                    x_tolerance: float = 0.0,
                    f_relative_tolerance: float = 0.0,
                    number_type: tf.DType = tf.float64):
        print('\nStarting L-BFGS optimization...')
        t1 = time.time()

        self.history_lbfgs = {'total': [], 'pde': [], 'dir': [], 'neu': [], 'data': [], 'val': []}
        init_weights = tf.concat([tf.reshape(v, [-1]) for v in self.model.trainable_variables], axis=0)

        try:
            optim_results = tfp.optimizer.lbfgs_minimize(
                self._function_factory(self.model, loss_fn, x_int, x_boundary, n_boundary, num_neumann, x_data, u_data, x_val, u_val, physics_fn, lambdas, number_type),
                initial_position=init_weights,
                max_iterations=max_iterations,
                num_correction_pairs=num_correction_pairs,
                tolerance=tolerance,
                x_tolerance=x_tolerance,
                f_relative_tolerance=f_relative_tolerance
            )

            # riapplicazione pesi
            sizes = [v.shape.num_elements() for v in self.model.trainable_variables]
            tensors = tf.split(optim_results.position, sizes)
            for v, t in zip(self.model.trainable_variables, tensors):
                v.assign(tf.reshape(t, v.shape))

            print(f'L-BFGS converged: {optim_results.converged.numpy()} after {optim_results.num_iterations.numpy()} iterations')

        except KeyboardInterrupt:
            print("\n[!] Ottimizzazione interrotta manualmente")
            ckpt_path = self._weights_path('lbfgs_checkpoint')
            if os.path.exists(ckpt_path):
                self.model.load_weights(ckpt_path)
                print(f"Weights loaded from checkpoint: {ckpt_path}")

        print(f'L-BFGS optimization finished in: {(time.time()-t1)/60:.2f} min')

        # salvataggio
        w_path, h_path = self.save_weights_and_history(self.model, self.history_lbfgs, 'decoder_lbfgs')
        
        data = {
            "optimizer": "L-BFGS",
            "execution_time_minutes": round((time.time() - t1) / 60, 2),
            "weights_file": w_path,
            "history_file": h_path,
            "lambdas": lambdas,
            "metrics": {
                "final_train_loss": float(self.history_lbfgs['total'][-1]) if self.history_lbfgs['total'] else -1.0,
                "final_val_loss": float(self.history_lbfgs['val'][-1]) if self.history_lbfgs['val'] else -1.0,
                "final_pde_loss": float(self.history_lbfgs['pde'][-1]) if self.history_lbfgs['pde'] else -1.0,
                "final_dir_loss": float(self.history_lbfgs['dir'][-1]) if self.history_lbfgs['dir'] else -1.0,
                "final_neu_loss": float(self.history_lbfgs['neu'][-1]) if self.history_lbfgs['neu'] else -1.0,
                "final_data_loss": float(self.history_lbfgs['data'][-1]) if self.history_lbfgs['data'] else -1.0
            }
        }
        
        self.update_experiment_log(phase_name="decoder_lbfgs", phase_data=data)


    def _function_factory(self, model, loss_fn, x_int, x_boundary, n_boundary, num_neumann, x_data, u_data, x_val, u_val, physics_fn, lambdas: Dict[str, float], number_type: tf.DType):
        # loss weights
        LAMBDA_PDE = tf.constant(lambdas['pde'], dtype=number_type)
        LAMBDA_DIR = tf.constant(lambdas['dir'], dtype=number_type)
        LAMBDA_NEU = tf.constant(lambdas['neu'], dtype=number_type)
        LAMBDA_DATA = tf.constant(lambdas['data'], dtype=number_type)

        step = tf.Variable(0, dtype=tf.int32, trainable=False)

        # XLA
        @tf.function(jit_compile=True)
        def compute_eval_xla(weights):
            sizes = [v.shape.num_elements() for v in model.trainable_variables]
            tensors = tf.split(weights, sizes)
            for v, t in zip(model.trainable_variables, tensors):
                v.assign(tf.reshape(t, v.shape))

            with tf.GradientTape() as tape:
                loss, lpde, ldir, lneu, ldata = loss_fn(
                    model, x_int, x_boundary, n_boundary, num_neumann, x_data, u_data,
                    physics_fn, LAMBDA_PDE, LAMBDA_DIR, LAMBDA_NEU, LAMBDA_DATA)

            grads = tape.gradient(loss, model.trainable_variables)
            grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

            u_pred_val = model(x_val)
            val_loss = tf.reduce_mean(tf.square(u_pred_val - u_val))

            return loss, grads_flat, lpde, ldir, lneu, ldata, val_loss

        def loss_grad(weights):
            loss, grads_flat, lpde, ldir, lneu, ldata, val_loss = compute_eval_xla(weights)

            def save_history(t_loss, t_pde, t_dir, t_neu, t_data, t_val):
                self.history_lbfgs['total'].append(t_loss.numpy())
                self.history_lbfgs['pde'].append(t_pde.numpy())
                self.history_lbfgs['dir'].append(t_dir.numpy())
                self.history_lbfgs['neu'].append(t_neu.numpy())
                self.history_lbfgs['data'].append(t_data.numpy())
                self.history_lbfgs['val'].append(t_val.numpy())
                return 0.0

            tf.py_function(save_history, [loss, lpde, ldir, lneu, ldata, val_loss], Tout=number_type)

            if tf.math.equal(step % 500, 0):
                tf.print(
                    "Call to loss_grad:", step,
                    "| Loss:", loss,
                    "| PDE:", lpde,
                    "| Dir:", ldir,
                    "| Neu:", lneu,
                    "| Data:", ldata,
                    "| Validation:", val_loss
                )

                def save_checkpoint_fn():
                    ckpt_path = self._weights_path('lbfgs_checkpoint')
                    model.save_weights(ckpt_path)
                    return 0.0
                tf.py_function(save_checkpoint_fn, [], Tout=number_type)

            step.assign_add(1)
            return loss, grads_flat

        return loss_grad