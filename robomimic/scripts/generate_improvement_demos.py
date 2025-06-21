import h5py
import numpy as np
import random
import matplotlib.pyplot as plt

def load_all_demos(hdf5_path):
    """
    Load all demos from HDF5 file once
    """
    demos = {}
    demo_lengths = {}
    env_args = None
    
    print("Loading all demos...")
    with h5py.File(hdf5_path, 'r') as f:
        # Load env_args if it exists
        if 'data' in f and hasattr(f['data'], 'attrs') and 'env_args' in f['data'].attrs:
            env_args = f['data'].attrs['env_args']
        
        for demo_id in f['data'].keys():
            demo_group = f['data'][demo_id]
            
            # Load demo data
            obs_dict = {}
            for obs_key in demo_group['obs'].keys():
                obs_dict[obs_key] = demo_group['obs'][obs_key][:]
            
            demo_data = {
                'actions': demo_group['actions'][:],
                'obs': obs_dict,
                'rewards': demo_group['rewards'][:],
                'dones': demo_group['dones'][:]
            }
            
            demos[demo_id] = demo_data
            demo_lengths[demo_id] = len(demo_data['actions'])  # Length of trajectory
    
    print(f"Loaded {len(demos)} demos")
    return demos, demo_lengths, env_args

def concatenate_demos(demos, demo_lengths, k, seed=None):
    """
    Concatenate k demos ensuring not all have the same length
    """
    if seed is not None:
        random.seed(seed)
    
    # Sample k demos, ensuring not all have same length
    k = min(k, len(demos))
    max_attempts = 100  # Prevent infinite loop
    attempts = 0
    
    while attempts < max_attempts:
        selected_demo_ids = random.sample(list(demos.keys()), k)
        selected_lengths = [demo_lengths[demo_id] for demo_id in selected_demo_ids]
        
        # Check if all lengths are the same
        if len(set(selected_lengths)) > 1:  # At least 2 different lengths
            break
        attempts += 1
    
    if attempts == max_attempts:
        print(f"Warning: Could not find {k} demos with different lengths after {max_attempts} attempts")
        # Fall back to random selection
        selected_demo_ids = random.sample(list(demos.keys()), k)
    
    # Sort by length (descending - longest to shortest)
    selected_demo_ids.sort(key=lambda x: demo_lengths[x], reverse=True)
    
    # Get individual lengths for plotting
    individual_lengths = [demo_lengths[demo_id] for demo_id in selected_demo_ids]
    
    # Concatenate demos
    # Handle obs dictionary concatenation
    obs_keys = demos[selected_demo_ids[0]]['obs'].keys()
    concatenated_obs = {}
    for obs_key in obs_keys:
        concatenated_obs[obs_key] = np.concatenate([demos[demo_id]['obs'][obs_key] for demo_id in selected_demo_ids])
    
    concatenated = {
        'actions': np.concatenate([demos[demo_id]['actions'] for demo_id in selected_demo_ids]),
        'obs': concatenated_obs,
        'rewards': np.concatenate([demos[demo_id]['rewards'] for demo_id in selected_demo_ids]),
        'dones': np.concatenate([demos[demo_id]['dones'] for demo_id in selected_demo_ids])
    }
    
    return concatenated, individual_lengths

def generate_concatenated_dataset(input_hdf5_path, output_hdf5_path, N, k, seed=None):
    """
    Generate N concatenated trajectories and save to HDF5
    
    Args:
        input_hdf5_path: Path to original dataset
        output_hdf5_path: Path to save concatenated dataset
        N: Number of concatenated trajectories to generate
        k: Number of demos to concatenate in each trajectory
        seed: Random seed
    
    Returns:
        List of returns for each concatenated trajectory (for plotting)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Load all demos once
    demos, demo_lengths, env_args = load_all_demos(input_hdf5_path)
    
    all_trajectory_lengths = []
    
    # Create output HDF5 file
    with h5py.File(output_hdf5_path, 'w') as out_f:
        data_group = out_f.create_group('data')
        data_group.attrs['total'] = 0
        # Store env_args if it exists
        if env_args is not None:
            data_group.attrs['env_args'] = env_args
        
        print(f"Generating {N} concatenated trajectories with k={k} demos each...")
        
        for i in range(N):
            # Generate concatenated demo (now fast since demos are in memory)
            concat_demo, individual_lengths = concatenate_demos(
                demos, demo_lengths, k, seed=None  # Use different seed for each trajectory
            )
            
            all_trajectory_lengths.append(individual_lengths)
            
            # Save to HDF5
            demo_group = data_group.create_group(f'demo_{i}')
            demo_group.attrs['num_samples'] = len(concat_demo['actions'])
            data_group.attrs['total'] += len(concat_demo['actions'])
            demo_group.create_dataset('actions', data=concat_demo['actions'])
            # demo_group.create_dataset('actions', data=concat_demo['obs']['actions'])
            
            # Save obs dictionary
            obs_group = demo_group.create_group('obs')
            for obs_key, obs_data in concat_demo['obs'].items():
                obs_group.create_dataset(obs_key, data=obs_data)
            
            demo_group.create_dataset('rewards', data=concat_demo['rewards'])
            demo_group.create_dataset('dones', data=concat_demo['dones'])
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{N} trajectories")
    
    print(f"Saved {N} concatenated trajectories to {output_hdf5_path}")
    return all_trajectory_lengths

def plot_average_lengths(all_trajectory_lengths, k):
    """
    Plot average lengths across k demos for all trajectories
    
    Args:
        all_trajectory_lengths: List of lists, each containing k lengths
        k: Number of demos per trajectory
    """
    # Convert to numpy array for easier manipulation
    lengths_array = np.array(all_trajectory_lengths)  # Shape: (N, k)
    
    # Calculate statistics
    mean_lengths = np.mean(lengths_array, axis=0)
    std_lengths = np.std(lengths_array, axis=0)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot individual trajectories (light lines)
    for i, trajectory_lengths in enumerate(all_trajectory_lengths[:min(50, len(all_trajectory_lengths))]):  # Show max 50 for clarity
        plt.plot(range(1, k+1), trajectory_lengths, 'lightgray', alpha=0.3, linewidth=0.5)
    
    # Plot average with error bars
    plt.errorbar(range(1, k+1), mean_lengths, yerr=std_lengths, 
                 color='blue', linewidth=2, marker='o', capsize=5, 
                 label=f'Average ± Std (N={len(all_trajectory_lengths)})')
    
    plt.xlabel('Demo Position in Concatenation (sorted by length)')
    plt.ylabel('Trajectory Length (steps)')
    plt.title(f'Lengths of {k} Demos in Concatenated Trajectories\n(Demos sorted by decreasing length)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, k+1))
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Total trajectories: {len(all_trajectory_lengths)}\n'
                         f'Demos per trajectory: {k}\n'
                         f'Mean length range: {mean_lengths[0]:.0f} → {mean_lengths[-1]:.0f} steps',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("improvement_data")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Average lengths by position: {mean_lengths}")
    print(f"Standard deviations: {std_lengths}")
    print(f"Length decrease from first to last: {mean_lengths[0] - mean_lengths[-1]:.0f} steps")

# Example usage
if __name__ == "__main__":
    # Parameters
    INPUT_HDF5 = './datasets/square.hdf5'  # Original dataset
    OUTPUT_HDF5 = './datasets/concatenated_square.hdf5'  # Output dataset
    N = 1000  # Number of concatenated trajectories to generate
    k = 4    # Number of demos to concatenate in each trajectory
    
    # Generate concatenated dataset
    trajectory_lengths = generate_concatenated_dataset(
        INPUT_HDF5, OUTPUT_HDF5, N=N, k=k, seed=42
    )
    
    # Plot the results
    plot_average_lengths(trajectory_lengths, k)
    
    # Verify the generated dataset
    print(f"\nVerifying generated dataset...")
    with h5py.File(OUTPUT_HDF5, 'r') as f:
        print(f"Number of demos in output: {len(f['data'].keys())}")
        demo_0 = f['data']['demo_0']
        print(f"First demo shapes:")
        for key in demo_0.keys():
            if key == 'obs':
                print(f"  obs:")
                for obs_key in demo_0['obs'].keys():
                    print(f"    {obs_key}: {demo_0['obs'][obs_key].shape}")
            else:
                print(f"  {key}: {demo_0[key].shape}")