import os
import random
import time
import numpy as np
import tensorflow as tf

from src.networks.DecoderPINN import PINN
from src.optimization.DecoderTrainer import DecoderTrainer
from src.data.decoder_dataloader import load_parametric_fem_data, load_parametric_fem_data_lbfgs
from src.data.collocation import generate_parametric_collocation_points
from src.physics.physics import get_M_dynamic
from src.visualization.historyplots import plot_adam_history, plot_lbfgs_history
from src.visualization.pdeplots import get_eval_grid, plot_parametric_pinn_solution
from src.utils.metrics import compute_test_metrics
from src.hyperparameters import hypers_uqvae as h

def train_and_evaluate_decoder(nx, ny, r_a, output_dir: str, result_dir: str, data_dir=None):
    tf.keras.backend.clear_session()
    
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    tf.keras.backend.set_floatx('float32')
    number_type = tf.float32

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    if data_dir is None:
        data_dir = f'data/parametric/{nx}x{ny}'

    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    test_path = os.path.join(data_dir, 'test.npz')

    scale_save_path = os.path.join(data_dir, 'u_scale.npy')

    (x_train_pinn, u_train_pinn), (x_val_pinn, u_val_pinn), (x_test_pinn, u_test_pinn), U_SCALE = load_parametric_fem_data(
        train_path=train_path, val_path=val_path, test_path=test_path,
        scale_save_path=scale_save_path, dtype=number_type
    )

    model = PINN(h.N_NEURONS_DECODER, U_SCALE)
    model(tf.zeros((1, 4)))

    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.8, staircase=False
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer.build(model.trainable_variables)

    def point_generator_fn():
        return generate_parametric_collocation_points(
            N_interior=h.N_INTERIOR_ADAM, N_boundary=h.N_BOUNDARY_ADAM, N_activation=h.N_ACTIVATION_ADAM,
            r_a = r_a, dtype=number_type
        )

    lambdas = {'pde': h.LAMBDA_PDE_ADAM, 'neu': h.LAMBDA_NEU_ADAM, 'dir': h.LAMBDA_DIR_ADAM, 'data': h.LAMBDA_DATA_ADAM}

    t0_adam = time.time()
    trainer = DecoderTrainer(model=model, output_dir=output_dir, result_dir=result_dir)
    trainer.train_adam(optimizer=optimizer, epochs=h.EPOCHS_DECODER_ADAM,
                       x_train=x_train_pinn, u_train=u_train_pinn,
                       x_val=x_val_pinn, u_val=u_val_pinn,
                       physics_fn=get_M_dynamic, lambdas=lambdas,
                       point_generator_fn=point_generator_fn,
                       batch_size=h.BATCH_SIZE_DECODER_ADAM,
                       patience=h.PATIENCE_DECODER_ADAM, number_type=number_type)
    t_adam = time.time() - t0_adam

    os.makedirs(result_dir, exist_ok=True)
    history_adam_path = os.path.join(result_dir, 'decoder_adam.loss.npz')
    np.savez(history_adam_path, **trainer.history_adam)
    plot_adam_history(trainer.history_adam, save_freq=10)

    del model
    del trainer
    tf.keras.backend.clear_session()

    tf.keras.backend.set_floatx('float64')
    number_type_lbfgs = tf.float64

    scale_load_path = os.path.join(data_dir, 'u_scale.npy')
    (x_train_lbfgs, u_train_lbfgs), (x_val_lbfgs, u_val_lbfgs), (x_test_lbfgs, u_test_lbfgs), U_SCALE = load_parametric_fem_data_lbfgs(
        train_path=train_path, val_path=val_path, test_path=test_path,
        scale_load_path=scale_load_path, n_lbfgs_data=25000,
        dtype=number_type_lbfgs
    )

    model_lbfgs = PINN(h.N_NEURONS_DECODER, U_SCALE)
    model_lbfgs(tf.zeros((1, 4), dtype=number_type_lbfgs))

    pretrained_weights = os.path.join(output_dir, 'decoder_adam.weights.h5')
    model_lbfgs.load_weights(pretrained_weights)
    print(f"Transfer Learning from Adam weights: {pretrained_weights}")

    x_int, x_boundary, n_boundary, num_neumann = generate_parametric_collocation_points(
        h.N_INTERIOR_LBFGS, h.N_BOUNDARY_LBFGS, h.N_ACTIVATION_LBFGS,r_a=r_a, dtype=number_type_lbfgs
    )

    lambdas_lbfgs = {'pde': h.LAMBDA_PDE_LBFGS, 'neu': h.LAMBDA_NEU_LBFGS, 'dir': h.LAMBDA_DIR_LBFGS, 'data': h.LAMBDA_DATA_LBFGS}

    t0_lbfgs = time.time()
    trainer_lbfgs = DecoderTrainer(model=model_lbfgs, output_dir=output_dir, result_dir=result_dir)
    trainer_lbfgs.train_lbfgs(x_int=x_int, x_boundary=x_boundary, n_boundary=n_boundary,
                              num_neumann=num_neumann, x_data=x_train_lbfgs, u_data=u_train_lbfgs,
                              x_val=x_val_lbfgs, u_val=u_val_lbfgs,
                              physics_fn=get_M_dynamic, lambdas=lambdas_lbfgs,
                              max_iterations=h.MAX_LBFGS_ITERATIONS,
                              num_correction_pairs=h.NUM_CORRECTION_PAIRS_LBFGS,
                              tolerance=h.TOLERANCE_LBFGS, x_tolerance=h.X_TOLERANCE_LBFGS,
                              f_relative_tolerance=h.F_RELATIVE_TOLERANCE_LBFGS,
                              number_type=number_type_lbfgs)
    t_lbfgs = time.time() - t0_lbfgs

    history_lbfgs_path = os.path.join(result_dir, 'decoder_lbfgs.loss.npz')
    np.savez(history_lbfgs_path, **trainer_lbfgs.history_lbfgs)
    plot_lbfgs_history(trainer_lbfgs.history_lbfgs)

    cx_test, cy_test = 0.5, 0.5
    X, Y, X_tensor = get_eval_grid(N=400, cx=cx_test, cy=cy_test, dtype=number_type_lbfgs)

    u_pred_flat = model_lbfgs(X_tensor).numpy() * U_SCALE
    U_PINN = u_pred_flat.reshape(400, 400) * 1000.0

    plot_parametric_pinn_solution(X, Y, U_PINN, cx_test, cy_test, scar_edge=0.2)

    data_test = np.load(test_path)
    
    values_clean = data_test['values_clean'] if 'values_clean' in data_test else data_test['values']
    
    l2_error, mae_error = compute_test_metrics(
        model=model_lbfgs, coords_test=data_test['coords'], values_test=values_clean, 
        U_SCALE=U_SCALE, dtype=number_type_lbfgs
    )

    return model_lbfgs, float(U_SCALE), l2_error, mae_error, t_adam, t_lbfgs