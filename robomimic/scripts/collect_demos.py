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
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument(
    "--num_demos", type=int, default=10, help="Number of demonstrations to record."
)

args_cli = parser.parse_args()


def main():
    """Collect demonstrations from the environment using RSL-RL policy."""
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    from robomimic_datacollector import Hdf5DataCollector
    data_collector = Hdf5DataCollector(
        env_name="tool_hang",
        directory_path=output_dir,
        filename=output_file_name,
        num_demos=args_cli.num_demos,
        # env_config= env_cfg.to_dict(),
    )
    device = TorchUtils.get_torch_device(True)
    config, _ = FileUtils.config_from_checkpoint(
        ckpt_path=args_cli.checkpoint, verbose=True
    )
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.checkpoint, device=device, verbose=True)
    # ckpt_dict = json.loads(ckpt_dict)
    env, _ = FileUtils.env_from_checkpoint(ckpt_path=args_cli.checkpoint, ckpt_dict=ckpt_dict)
    
    # env = EnvUtils.wrap_env_from_config(env, config)
    obs = env.reset()
    data_collector.reset()
    reward, done = np.zeros((1, 1)), np.zeros((1, 1))
    policy.start_episode(resets=done)
    success_rates = []
    step = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while not data_collector.is_stopped():
            action = policy(obs)
            action = TensorUtils.to_numpy(action)
            for key in obs.keys():
                unstacked_obs = env.obs_history[key][-1][0]
                data_collector.add(
                    f"obs/{key}", unstacked_obs.reshape(1, -1)
                )
            data_collector.add(
                "actions", action.reshape(1, -1)
            )
            data_collector.add(
                "obs/rewards",
                reward
            )
            data_collector.add(
                "obs/dones",
                done.astype(np.float32)
            )
            data_collector.add(
                "rewards",
                reward
            )
            data_collector.add(
                "dones",
                done.astype(np.float32)
            )

            obs, reward, done, infos = env.step(action)
            step += 1
            if step > 400 or infos["is_success"]['task']:
                done = True
            
            reward = np.array([reward]).reshape(1, -1)
            done = np.array([done]).reshape(1, -1)
            if done.any():
                success_rates.append(infos["is_success"]['task'])
                data_collector.flush()
                obs = env.reset()
                reward, done = np.zeros((1, 1)), np.zeros((1, 1))
                policy.start_episode(resets=done)
                step = 0
    print(f"Success rate: {np.mean(success_rates)}")
    # save the dataset


if __name__ == "__main__":
    # run the main function
    main()