# test_dataset.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Optional: for progress bars
import os

from homework4 import CNP

# --- Parameters ---
DATASET_PATH = "dataset_200.npz"
MODEL_LOAD_PATH = "best_cnmp_model.pt" 
NUM_MSE_TESTS = 100
HIDDEN_SIZE = 128 
NUM_HIDDEN_LAYERS = 3 
MAX__POINTS = 50 

def load_data(path):
    data = np.load(path, allow_pickle=True)
    trajectories = data['data'] 
    print(f"Loaded {len(trajectories)} trajectories from {path}")
    return list(trajectories)


def evaluate_single_trajectory_mse(model, trajectory_raw, max_context, max_target):
    model.eval()

    with torch.no_grad():
        traj_len = trajectory_raw.shape[0]
        t_vals = trajectory_raw[:, 0:1]         
        target_vals = trajectory_raw[:, 1:5]    # Shape (traj_len, 4) -> ey, ez, oy, oz
        h_vals = trajectory_raw[:, 5:6]         # Shape (traj_len, 1)

        traj_reordered = np.concatenate([t_vals, h_vals, target_vals], axis=1)
        traj_tensor = torch.tensor(traj_reordered, dtype=torch.float32).unsqueeze(0)

        n_context = np.random.randint(1,  max_context )
        n_target = np.random.randint(1,  max_target )

        all_indices = np.arange(traj_len)
        context_indices = np.random.choice(all_indices, n_context, replace=False)
        target_indices = np.random.choice(all_indices, n_target, replace=False)

        d_x = model.d_x 
        d_y = model.d_y 

        observation = traj_tensor[:, context_indices, :]      
        target_query = traj_tensor[:, target_indices, :d_x]  
        target_truth = traj_tensor[:, target_indices, d_x:]  

        if n_target == 1 and len(target_truth.shape) == 2:
             target_truth = target_truth.unsqueeze(1)

        mean_pred, _ = model(observation, target_query) 

        squared_error = (mean_pred - target_truth)**2
        mse_per_dim = torch.mean(squared_error, dim=1).squeeze(0) # Average over n_target points

        mse_ee = torch.mean(mse_per_dim[:2]).item() # First 2 dims are end-effector
        mse_obj = torch.mean(mse_per_dim[2:]).item() # Last 2 dims are object

        return mse_ee, mse_obj


if __name__ == "__main__":

    all_trajectories = load_data(DATASET_PATH)

    model = CNP(in_shape=(2, 4), hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS)


    model.load_state_dict(torch.load(MODEL_LOAD_PATH))

    print(f"\n--- Running {NUM_MSE_TESTS} MSE Evaluation Tests ---")
    ee_mses = []
    obj_mses = []
    successful_evals = 0
    pbar = tqdm(total=NUM_MSE_TESTS, desc="MSE Evaluation")
    while successful_evals < NUM_MSE_TESTS:
        # Randomly select one trajectory from the loaded dataset for this test
        test_traj_idx = np.random.randint(0, len(all_trajectories))
        test_traj = all_trajectories[test_traj_idx]

        # Evaluate on the selected trajectory
        mse_ee, mse_obj = evaluate_single_trajectory_mse(
            model, test_traj, max_context=MAX__POINTS, max_target=MAX__POINTS
        )

        if mse_ee is not None and mse_obj is not None:
            ee_mses.append(mse_ee)
            obj_mses.append(mse_obj)
            successful_evals += 1
            pbar.update(1)
    pbar.close()
    
    if not ee_mses or not obj_mses:
         print("Error: No successful evaluations were completed.")
         exit()

    # --- Calculate and Print Results ---
    mean_mse_ee = np.mean(ee_mses)
    std_mse_ee = np.std(ee_mses)
    mean_mse_obj = np.mean(obj_mses)
    std_mse_obj = np.std(obj_mses)

    print("\n--- Evaluation Results ---")
    print(f"Number of successful tests: {len(ee_mses)}")
    print(f"End-Effector MSE: Mean={mean_mse_ee:.6f}, Std={std_mse_ee:.6f}")
    print(f"Object MSE:       Mean={mean_mse_obj:.6f}, Std={std_mse_obj:.6f}")

    # --- Plotting MSE Results ---
    labels = ['CNMP (Trained)']
    ee_means = [mean_mse_ee]
    ee_stds = [std_mse_ee]
    obj_means = [mean_mse_obj]
    obj_stds = [std_mse_obj]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))

    rects1 = ax.bar(x - width/2, ee_means, width, yerr=ee_stds,
                    label='End-Effector MSE', color='royalblue', capsize=5, alpha=0.9)
    rects2 = ax.bar(x + width/2, obj_means, width, yerr=obj_stds,
                    label='Object MSE', color='darkorange', capsize=5, alpha=0.9)

    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(f'CNMP Prediction MSE ({len(ee_mses)} Tests on Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()

    plot_filename = "test_dataset_results_plot.png"

    plt.savefig(plot_filename)
    print(f"Results plot saved to {plot_filename}")

    plt.show()

    print("\nTesting finished.")