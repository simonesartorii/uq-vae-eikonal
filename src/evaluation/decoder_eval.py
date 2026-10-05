import os
import time
import numpy as np
import tensorflow as tf
from fenics import *

from src.physics import constants as c
from src.networks.DecoderPINN import PINN
from src.hyperparameters import hypers_uqvae as h

def solve_eikonal_fem(nx, ny, r_a, cx=0.5, cy=0.5):
    """
    FEM solver on a nx x ny grid, with scar center in (cx, cy)
    """
    set_log_level(40) 
    
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'CG', 1)
    
    b_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    b_markers.set_all(0)
    
    for f in facets(mesh):
        if f.exterior():
            x, y = f.midpoint().x(), f.midpoint().y()
            if np.isclose(np.sqrt(x**2 + y**2), 0.0, atol=r_a):
                b_markers[f] = 1
                
    bc = DirichletBC(V, Constant(0.0), b_markers, 1)
    
    v_teo = 62.0 * np.sqrt(1.529)
    u_initial = Expression(f'sqrt(pow(x[0] - x0, 2) + pow(x[1] - y0, 2)) / {v_teo}', degree=1, x0=0.0, y0=0.0)
    
    M_tensor = Expression('(x[0] >= cx - 0.1 && x[0] <= cx + 0.1 && x[1] >= cy - 0.1 && x[1] <= cy + 0.1) ? m_scar : m_normal',
                          degree=0, cx=cx, cy=cy, m_scar=c.M_SCAR, m_normal=c.M_VAL)
    
    psi, v = Function(V), TestFunction(V)
    eps, c0 = Constant(c.EPS_VAL), Constant(c.C0_VAL)
    
    grad_psi = grad(psi)
    M_grad_psi = M_tensor * grad_psi
    F = (c0 * sqrt(inner(grad_psi, M_grad_psi)) * v * dx + eps * inner(M_grad_psi, grad(v)) * dx - Constant(1.0) * v * dx)
    J = derivative(F, psi)
    
    pb = NonlinearVariationalProblem(F, psi, bc, J)
    solver = NonlinearVariationalSolver(pb)
    prm = solver.parameters
    
    prm['newton_solver']['linear_solver'] = 'gmres'
    prm['newton_solver']['preconditioner'] = 'amg'
    prm['newton_solver']['absolute_tolerance'] = 1E-8
    prm['newton_solver']['relative_tolerance'] = 1E-7
    prm['newton_solver']['error_on_nonconvergence'] = False

    psi.interpolate(u_initial)
    solver.solve()
    
    return psi

def compute_convergence_rates(resolutions=[100, 200, 400, 800], cx_test=0.5, cy_test=0.5):
    r_a = 1.0 / resolutions[0]
    print(f"cx={cx_test}, cy={cy_test}, r_a={r_a}...")
    solutions = {}
    
    for N in resolutions:
        t0 = time.time()
        print(f"Solving FEM grid {N}x{N}...", end="", flush=True)
        u_h = solve_eikonal_fem(N, N, r_a=r_a, cx=cx_test, cy=cy_test)
        solutions[N] = u_h
        print(f" Completed in {time.time()-t0:.2f} s")
        
    u_ref = solutions[resolutions[-1]] 
    eval_resolutions = resolutions[:-1]
    
    errors_L2, errors_H1, hs = [], [], []
    
    for N in eval_resolutions:
        u_h = solutions[N]
        h = 1.0 / N
        hs.append(h)
        
        e_L2 = errornorm(u_ref, u_h, norm_type='L2', degree_rise=1)
        e_H1 = errornorm(u_ref, u_h, norm_type='H1', degree_rise=1) 
        
        errors_L2.append(e_L2)
        errors_H1.append(e_H1)
        
    eoc_L2, eoc_H1 = [np.nan], [np.nan]
    
    for i in range(1, len(hs)):
        rate_L2 = np.log(errors_L2[i] / errors_L2[i-1]) / np.log(hs[i] / hs[i-1])
        rate_H1 = np.log(errors_H1[i] / errors_H1[i-1]) / np.log(hs[i] / hs[i-1])
        eoc_L2.append(rate_L2)
        eoc_H1.append(rate_H1)
        
    print("\nFEM Errors:")
    print("-" * 75)
    print(f"{'N':<10} | {'h':<10} | {'Error L2':<15} | {'EOC L2':<10} | {'Error H1':<15} | {'EOC H1':<10}")
    print("-" * 75)
    
    for i, N in enumerate(eval_resolutions):
        h_str = f"{hs[i]:.4f}"
        eL2_str = f"{errors_L2[i]:.4e}"
        eH1_str = f"{errors_H1[i]:.4e}"
        rL2_str = f"{eoc_L2[i]:.2f}" if not np.isnan(eoc_L2[i]) else "-"
        rH1_str = f"{eoc_H1[i]:.2f}" if not np.isnan(eoc_H1[i]) else "-"
        
        print(f"{N:<10} | {h_str:<10} | {eL2_str:<15} | {rL2_str:<10} | {eH1_str:<15} | {rH1_str:<10}")
        
    return solutions, eval_resolutions, hs, errors_L2, eoc_L2, errors_H1, eoc_H1


def evaluate_pinn_accuracy(base_data_dir='data/multiresolution',
                           base_model_dir='models/error_analysis',
                           resolutions=["100x100", "200x200", "400x400", "800x800"],
                           gt_res="800x800"):
    """
    Evaluates PINN architectures across resolutions against a single ground truth dataset.
    """
    tf.keras.backend.set_floatx('float64')
    number_type = tf.float64
    
    gt_path = os.path.join(base_data_dir, gt_res, 'test.npz')
    data_test_gt = np.load(gt_path)
    
    coords_test = data_test_gt['coords']
    values_clean_800 = data_test_gt['values_clean']
    
    pinn_resolutions = []
    pinn_l2_errors = []
    
    print(f"\nRelative L2 Error (PINN vs CLEAN Ground Truth {gt_res}):")
    print("-" * 85)
    
    for resolution in resolutions:
        tf.keras.backend.clear_session()
            
        res_data_dir = os.path.join(base_data_dir, resolution)
        res_model_dir = os.path.join(base_model_dir, resolution)
       
        scale_path = os.path.join(res_data_dir, 'u_scale.npy')
        if not os.path.exists(scale_path):
            print(f"[!] PINN trained on FEM {resolution:<10} | Missing {scale_path}")
            continue
            
        u_scale_model = float(np.load(scale_path)[0])
        
        model_eval = PINN(h.N_NEURONS_DECODER, u_scale_model)
        model_eval(tf.zeros((1, 4), dtype=number_type))
        
        weight_path = os.path.join(res_model_dir, 'decoder_lbfgs.weights.h5')
        if not os.path.exists(weight_path):
            weight_path = os.path.join(res_model_dir, 'decoder_adam.weights.h5')
    
        try:
            model_eval.load_weights(weight_path)
        except Exception as e:
            print(f"[!] PINN trained on FEM {resolution:<10} | Error loading weights: {e}")
            continue
        
        coords_tf = tf.cast(coords_test, dtype=number_type)
        u_pred_norm = model_eval(coords_tf)
        u_pred = u_pred_norm.numpy() * u_scale_model
        
        diff = u_pred.flatten() - values_clean_800.flatten()
        norm_diff = np.linalg.norm(diff)
        norm_clean = np.linalg.norm(values_clean_800.flatten())
        
        l2_err_pinn = (norm_diff / norm_clean) * 100.0
        
        pinn_resolutions.append(resolution)
        pinn_l2_errors.append(l2_err_pinn)
        
        print(f"PINN trained on FEM {resolution:<10} | L2 Rel Error vs Clean {gt_res}: {l2_err_pinn:.4f} %")
        
    return pinn_resolutions, pinn_l2_errors