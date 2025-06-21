# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-2025, The Octi Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import sys
import os
import torch
from tqdm import tqdm
import json

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils

import numpy as np
import gymnasium as gym

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations from the environment using robomimic policy.")
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the checkpoint file of the policy to use for collecting demos."
)
parser.add_argument(
    "--improvement_checkpoint", type=str, required=True, help="Path to the checkpoint file of the policy to use for collecting improvement demos."
)
parser.add_argument(
    "--num_eval_episodes", type=int, default=10, help="Number of demonstrations to eval."
)
parser.add_argument(
    "--num_sub_episodes", type=int, default=4, help="Number of sub-episodes in each episode."
)
parser.add_argument(
    "--transformer", action="store_true",
    help="Use transformer policy for improvement.",
)

args_cli = parser.parse_args()


def main():
    """Collect demonstrations from the environment using RSL-RL policy."""
    device = TorchUtils.get_torch_device(True)
    config, _ = FileUtils.config_from_checkpoint(
        ckpt_path=args_cli.checkpoint, verbose=True
    )
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.checkpoint, device=device, verbose=True)
    improvement_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.improvement_checkpoint, device=device, verbose=True)
    # create environment
    # ckpt_dict["config"] = json.loads(ckpt_dict["config"])
    # ckpt_dict["config"]["train"]["frame_stack"] = 1
    env, _ = FileUtils.env_from_checkpoint(ckpt_path=args_cli.checkpoint, ckpt_dict=ckpt_dict)
    if args_cli.transformer:
        from robomimic.envs.wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, num_frames=1024)

    all_success_rates = []
    step = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        # env = EnvUtils.wrap_env_from_config(env, config)
        # obs = env.reset()
        # reward, done = np.zeros((1, 1)), np.zeros((1, 1))
        policy.start_episode(resets=np.ones((1, 1)))
        improvement_policy.start_episode(resets=np.ones((1, 1)))

        for _ in tqdm(range(args_cli.num_eval_episodes), desc="Evaluating episodes"):
            success_rates = []
            context_length = 0
            for sub_episode in range(args_cli.num_sub_episodes):
                context_length = min(context_length+1, 1024)
                obs = env.reset()
                reward, done = np.zeros((1, 1)), np.zeros((1, 1))
                step = 0
                while True:
                    if args_cli.transformer:
                        base_obs = dict()
                        for key in obs.keys(): # 1024 x 2 x obs
                            base_obs[key] = obs[key][-1] # 1024 x obs
                        im_obs = dict()
                        for key in obs.keys():
                            im_obs[key] = obs[key][-context_length:, -1] # context_length x obs
                        im_obs["rewards"] = np.zeros((context_length, 1))
                        im_obs["dones"] = np.zeros((context_length, 1))
                    else:
                        base_obs = obs
                        im_obs = dict()
                        for key in obs.keys():
                            im_obs[key] = env.obs_history[key][-context_length:, 0].flatten()
                        im_obs["rewards"] = reward.flatten()
                        im_obs["dones"] = done.flatten()
                    
                    base_action = policy(base_obs)
                    im_action = improvement_policy(im_obs)
                    action = base_action if sub_episode == 0 else im_action
                    action = TensorUtils.to_numpy(action)

                    obs, reward, done, infos = env.step(action)
                    if step > 400 or infos["is_success"]['task']:
                        done = True
                    reward = np.array([reward]).reshape(1, -1)
                    done = np.array([done]).reshape(1, -1)
                    step += 1

                    if done.any():
                        success_rates.append(infos["is_success"]['task'])
                        break
            all_success_rates.append(np.array(success_rates))

    all_success_rates = np.array(all_success_rates)
    print(f"Average success rate over {args_cli.num_eval_episodes} episodes: {np.mean(all_success_rates)}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    initial_success_rate = all_success_rates[:, 0].mean()
    mean_success_rate = all_success_rates.mean(axis=0)
    std_success_rate = all_success_rates.std(axis=0)
    plt.plot(range(args_cli.num_sub_episodes), mean_success_rate, label='Mean Success Rate', marker='o')
    plt.fill_between(range(args_cli.num_sub_episodes), 
                     mean_success_rate - std_success_rate, 
                     mean_success_rate + std_success_rate, 
                     alpha=0.2, label='Std Dev')
    plt.axhline(y=initial_success_rate, color='r', linestyle='--', label='Initial Success Rate')
    plt.xticks(range(args_cli.num_sub_episodes))
    plt.xlim(0, args_cli.num_sub_episodes - 1)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('success_rate_per_sub_episode.png')
    plt.show()

if __name__ == "__main__":
    # run the main function
    main()