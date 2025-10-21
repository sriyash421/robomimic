#!/bin/bash

sbatch train_job.sh robomimic/exps/diffusion_policy_h4_image.json test-image-square-dp-h4 datasets/square/ph/image_v15.hdf5

sbatch train_job.sh robomimic/exps/mean_flow_policy_h4_image.json test-image-square-dp-h4 datasets/square/ph/image_v15.hdf5

sbatch train_job.sh robomimic/exps/mean_flow_policy_h8_image.json test-image-square-dp-h8 datasets/square/ph/image_v15.hdf5

sbatch train_job.sh robomimic/exps/diffusion_policy_h8_image.json test-image-square-dp-h8 datasets/square/ph/image_v15.hdf5