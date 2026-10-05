import os
import json
import time
import numpy as np
from fenics import *
from src.physics import constants as c

# helper functions
def _build_mesh_and_bc(N, r_a):
    """Mesh, function space, boundary conditions, initial conditions."""
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'CG', 1)

    b_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    b_markers.set_all(0)
    for f in facets(mesh):
        if f.exterior():
            x, y = f.midpoint().x(), f.midpoint().y()
            if np.isclose(np.sqrt(x**2 + y**2), 0.0, atol=r_a):
                b_markers[f] = 1
    bc = DirichletBC(V, Constant(0.0), b_markers, 1)

    v_teo = c.C0_VAL * np.sqrt(c.M_VAL)
    u_initial = Expression(f'sqrt(pow(x[0] - x0, 2) + pow(x[1] - y0, 2)) / {v_teo}',
                            degree=1, x0=0.0, y0=0.0)
    return V, bc, u_initial


def _build_solver(V, bc, M_tensor):
    """Solver for a given M_tensor"""
    psi, v = Function(V), TestFunction(V)
    eps, c0 = Constant(c.EPS_VAL), Constant(c.C0_VAL)
    grad_psi = grad(psi)
    M_grad_psi = M_tensor * grad_psi
    F = (c0 * sqrt(inner(grad_psi, M_grad_psi)) * v * dx
            + eps * inner(M_grad_psi, grad(v)) * dx - Constant(1.0) * v * dx)
    J = derivative(F, psi)

    pb = NonlinearVariationalProblem(F, psi, bc, J)
    solver = NonlinearVariationalSolver(pb)
    prm = solver.parameters
    prm['newton_solver']['linear_solver'] = 'gmres'
    prm['newton_solver']['preconditioner'] = 'amg'
    prm['newton_solver']['absolute_tolerance'] = 1E-8
    prm['newton_solver']['relative_tolerance'] = 1E-7
    prm['newton_solver']['error_on_nonconvergence'] = False

    return psi, solver


def generate_fem_data(N=400, r_a=0.0025, base_dir='data/fixed_scar'):
    """Generates FEM data with non constant M_tensor"""
    os.makedirs(base_dir, exist_ok=True)
    set_log_level(40)

    V, bc, u_initial = _build_mesh_and_bc(N, r_a)
    coords_fem = V.tabulate_dof_coordinates()

    M_tensor = Expression(
        '(x[0] >= x_min && x[0] <= x_max && x[1] >= y_min && x[1] <= y_max) ? m_scar : m_normal',
        degree=0, x_min=c.SCAR_X_MIN, x_max=c.SCAR_X_MAX,
        y_min=c.SCAR_Y_MIN, y_max=c.SCAR_Y_MAX,
        m_scar=c.M_SCAR, m_normal=c.M_VAL)

    psi, solver = _build_solver(V, bc, M_tensor)
    psi.interpolate(u_initial)

    n_it, converged = solver.solve()
    print(f"Converged in {n_it} iterations." if converged else f"Did not converge in {n_it} iterations")

    values_fem = psi.vector().get_local().reshape(-1, 1)
    dataset_path = f'{base_dir}/dataset.npz'
    np.savez(dataset_path, coords=coords_fem, values=values_fem)
    print(f"[DATA] Saved {dataset_path}")

    x_test = np.linspace(0.0, 1.0, N)
    y_test = np.linspace(0.0, 1.0, N)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    X_test_flattened = np.hstack((X_test.flatten()[:, None], Y_test.flatten()[:, None]))

    u_fem_grid = np.array([psi(pt) for pt in X_test_flattened])
    grid_path = f'{base_dir}/fem_grid.npy'
    np.save(grid_path, u_fem_grid)
    print(f"[DATA] Saved {grid_path}")

    return dataset_path


def generate_parametric_fem_data(resolutions=(100, 200, 400, 800),
                                 n_sim=1000, n_points_save=1020,
                                 seed=42, r_a=0.01,
                                 base_dir='data/parametric',
                                 save_shared_data=True,
                                 save_grid=False):
    """
    Generates parametric FEM simulations for one or more grid resolutions, handles handles train/val/test split.
    
    Args:
        resolutions: list of grid resolutions to generate
        n_sim: number of FEM simulations
        n_points_to_save: number of points to save for each simulation
        r_a: activation radius, used to handle Dirichlet BC 
        save_shared_data: if True, the function also saves shared data (centers.npz, evaluation_points.npz) used for the error analysis pipline          
        save_grid: if True, the function also saves the solution on a NxN uniform grid
    """
    os.makedirs(base_dir, exist_ok=True)
    set_log_level(40)
    np.random.seed(seed)

    # centers (fixed candidates) 
    cx_vals = np.random.uniform(0.2, 0.8, n_sim)
    cy_vals = np.random.uniform(0.2, 0.8, n_sim)
    candidate_centers = list(set([(round(cx, 3), round(cy, 3))
                                    for cx, cy in zip(cx_vals, cy_vals)]))
    np.random.shuffle(candidate_centers)

    # spatial coordinates (sampled from coarsest mesh)
    N_base = resolutions[0]
    mesh_base = UnitSquareMesh(N_base, N_base)
    V_base = FunctionSpace(mesh_base, 'CG', 1)
    coords_base = V_base.tabulate_dof_coordinates()
    point_idx = np.random.choice(len(coords_base), n_points_save, replace=False)
    fixed_points = coords_base[point_idx]

    # dictionary to store valid solutions: results[(cx, cy)][N]
    results = {center: {} for center in candidate_centers}
    if save_grid:
        results_grid = {center: {} for center in candidate_centers}

    time_per_res = {}

    print("\n" + "="*85)
    print("PARAMETRIC DATA GENERATION")
    print(f"Checking convergence for {len(candidate_centers)} candidate centers")
    print("="*85)

    # solvers loop (each resolution is handled separately)
    for N in resolutions:
        print(f"\n[INFO] RESOLUTION: {N}x{N}")
        t0_res = time.time()

        if save_grid:
            x_test = np.linspace(0.0, 1.0, N)
            y_test = np.linspace(0.0, 1.0, N)
            X_test, Y_test = np.meshgrid(x_test, y_test)
            X_grid_flat = np.hstack((X_test.flatten()[:, None], Y_test.flatten()[:, None]))
            
        M_tensor = Expression(
                '(x[0] >= cx - 0.1 && x[0] <= cx + 0.1 && x[1] >= cy - 0.1 && x[1] <= cy + 0.1) ? m_scar : m_normal',
                degree=0, cx=0.5, cy=0.5, m_scar=c.M_SCAR, m_normal=c.M_VAL)
        V, bc, u_initial = _build_mesh_and_bc(N, r_a)
        psi, solver = _build_solver(V, bc, M_tensor)

        # only iterate on centers that didn't fail in the previous resolution 
        active_centers = [c for c in candidate_centers if c in results]
        
        for idx, (cx, cy) in enumerate(active_centers):
            M_tensor.cx, M_tensor.cy = cx, cy
            psi.interpolate(u_initial)
            n_it, converged = solver.solve()
            
            if not converged:
                print(f"  [SKIPPED] cx={cx:.3f}, cy={cy:.3f} | Newton solver failed. Dropping from all grids.")
                del results[(cx, cy)]
                continue
            
            # evaluate solution exactly at the fixed points
            results[(cx, cy)][N] = np.array([psi(pt) for pt in fixed_points], dtype=np.float32)
            
            if save_grid:
                results_grid[(cx, cy)][N] = np.array([psi(pt) for pt in X_grid_flat], dtype=np.float32).reshape(-1, 1)

            if idx > 0 and idx % 100 == 0:
                print(f"  Simulation {idx}: cx = {cx:.3f}, cy = {cy:.3f}")
        
        t_res = time.time() - t0_res
        time_per_res[N] = t_res
        print(f"  -> Generation for {N}x{N} completed in {t_res:.2f} s")

    # extract only the centers in results 
    kept_centers = [c for c in candidate_centers if c in results]
    n_kept = len(kept_centers)
    
    print("\n" + "-" * 85)
    print(f"Valid centers across all resolutions: {n_kept} / {len(candidate_centers)}")
    print("-" * 85)

    # train/val/test split
    n_train = int(0.8 * n_kept)
    n_val = int(0.1 * n_kept)
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n_kept)

    centers_arr = np.array(kept_centers)

    # save shared data
    if save_shared_data:
        np.savez(f'{base_dir}/evaluation_points.npz', points=fixed_points)
        np.savez(f'{base_dir}/centers.npz', train=centers_arr[idx_train], val=centers_arr[idx_val], test=centers_arr[idx_test])

    # saving per resolution
    for N in resolutions:
        save_dir = f'{base_dir}/{N}x{N}'
        os.makedirs(save_dir, exist_ok=True)

        values_arr = np.array([results[c][N] for c in kept_centers], dtype=np.float32)
        if save_grid:
            values_grid_arr = np.array([results_grid[cnt][N] for cnt in kept_centers], dtype=np.float32)
            x_test = np.linspace(0.0, 1.0, N)
            y_test = np.linspace(0.0, 1.0, N)
            X_test, Y_test = np.meshgrid(x_test, y_test)
            X_grid_flat = np.hstack((X_test.flatten()[:, None], Y_test.flatten()[:, None]))
            n_pts_grid = N * N

        for split_name, idx in [('train', idx_train), ('val', idx_val), ('test', idx_test)]:
            centers_split = centers_arr[idx]
            values_split = values_arr[idx]

            n_split = len(centers_split)
            coords_out = np.zeros((n_split * n_points_save, 4), dtype=np.float32)
            values_clean_out = np.zeros((n_split * n_points_save, 1), dtype=np.float32)

            for j, (cx, cy) in enumerate(centers_split):
                sl = slice(j * n_points_save, (j + 1) * n_points_save)
                coords_out[sl, 0] = fixed_points[:, 0]
                coords_out[sl, 1] = fixed_points[:, 1]
                coords_out[sl, 2] = cx
                coords_out[sl, 3] = cy
                values_clean_out[sl, 0] = values_split[j]

            np.savez(f'{save_dir}/{split_name}.npz',coords=coords_out, values_clean=values_clean_out)

            if save_grid:
                values_grid_split = values_grid_arr[idx]
                coords_grid_out = np.zeros((n_split * n_pts_grid, 4), dtype=np.float32)
                values_grid_clean_out = np.zeros((n_split * n_pts_grid, 1), dtype=np.float32)
                
                for j, (cx, cy) in enumerate(centers_split):
                    sl = slice(j * n_pts_grid, (j + 1) * n_pts_grid)
                    coords_grid_out[sl, 0] = X_grid_flat[:, 0]
                    coords_grid_out[sl, 1] = X_grid_flat[:, 1]
                    coords_grid_out[sl, 2] = cx
                    coords_grid_out[sl, 3] = cy
                    values_grid_clean_out[sl, 0] = values_grid_split[j].flatten()
                    
                np.savez(f'{save_dir}/{split_name}_grid.npz', coords=coords_grid_out, values_clean=values_grid_clean_out)
                
        print(f"[DATA] Saved in {save_dir} | Train: {n_train} | Val: {n_val} | Test: {n_kept - n_train - n_val}")
    
    if save_shared_data:
         with open(f'{base_dir}/generation_times.json', 'w') as f:
                json.dump({str(k): v for k, v in time_per_res.items()}, f, indent=4)
        
    print("\n" + "="*85)
    print(f"[DONE] Generation completed. Data saved in {base_dir}")
    print("="*85 + "\n")
    return base_dir, time_per_res


def generate_noise(resolutions=[100, 200, 400, 800], base_dir='data/parametric',
                    noise_level_relative=0.02, seed=42):
    """
    Generates a single noise realization for each (simulation, point) pair, shared across all resolutions.
    """
    np.random.seed(seed)
    ref = resolutions[-1]

    signal_values = []
    for split_name in ['train']:
        data = np.load(f'{base_dir}/{ref}x{ref}/{split_name}.npz')
        signal_values.append(data['values_clean'])
    signal_scale = np.max(np.abs(np.concatenate(signal_values, axis=0)))
    noise_std_physical = noise_level_relative * signal_scale

    print(f"Reference grid: {ref}x{ref}")
    print(f"Signal scale (max|values_clean|): {signal_scale:.6f}")
    print(f"Noise std (physical, {noise_level_relative*100:.1f}% of signal): {noise_std_physical:.6f}")

    noise_splits = {}
    for split_name in ['train', 'val', 'test']:
        n_rows = np.load(f'{base_dir}/{ref}x{ref}/{split_name}.npz')['coords'].shape[0]
        noise_splits[split_name] = np.random.normal(
            0.0, noise_std_physical, size=(n_rows, 1)).astype(np.float32)
 
    ref_coords = np.load(f'{base_dir}/{ref}x{ref}/train.npz')['coords']
    cx0, cy0 = ref_coords[0, 2], ref_coords[0, 3]
    n_pts = len(ref_coords)
    for i in range(1, len(ref_coords)):
        if ref_coords[i, 2] != cx0 or ref_coords[i, 3] != cy0:
            n_pts = i
            break

    noise_kept = np.concatenate(
        [noise_splits[s].reshape(-1, n_pts) for s in ['train', 'val', 'test']], axis=0)

    np.savez(f'{base_dir}/noise.npz',
                noise=noise_kept,
                noise_std_physical=noise_std_physical,
                signal_scale=signal_scale,
                noise_level_relative=noise_level_relative,
                seed=seed)
    print(f"[OK] Saved {base_dir}/noise.npz")

    for split_name in ['train', 'val', 'test']:
        for N in resolutions:
            path_N = f'{base_dir}/{N}x{N}/{split_name}.npz'
            data_N = np.load(path_N)
            np.savez(path_N,
                        coords=data_N['coords'],
                        values_clean=data_N['values_clean'],
                        noise=noise_splits[split_name])
            print(f"[OK] Noise updated in {path_N}")