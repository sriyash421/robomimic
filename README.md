# RoboCasa Policy Learning Repo

This is the official policy learning repo accompanying the [RoboCasa](https://robocasa.ai/) release. This repo is based on top of robomimic, with modifications to train on RoboCasa datasets.

-------
## Installation
```
pip install robosuite==1.4.1
git clone https://github.com/ARISE-Initiative/robomimic -b robocasa
cd robomimic
pip install -e .
```

-------
## Documentatation
Please refer to the policy learning [documentation page](https://robocasa.ai/docs/use_cases/policy_learning.html) for information on training and evaluating policies.


-------

# Improvement Learning

## Step 1. Get the datasets


## Step 2. Training base diffusion policy

```bash
python robomimic/scripts/train.py \
    --config robomimic/exps/templates/diffusion_policy.json --name square \              
    --dataset ../jaxrl_m/jaxrl_m/envs/demos/robomimic_datasets/square/ph/low_dim_v141.hdf5
```

## Step 3. Collect demos
```
python robomimic/scripts/collect_demos.py --checkpoint diffusion_policy_trained_models/square/20250612142846/models/model_epoch_100.pth --dataset_file datasets/square.hdf5 --num_demos 200
```

## Step 4. Generate ranked improvement data

- Currently ranked on the basis of inverse of trajectory length
- Paths are hardcoded in the file

```
python robomimic/scripts/generate_improvement_demos.py
```

## Step 5. Train improvement model
```
python robomimic/scripts/train.py \
    --config robomimic/exps/sriyash-im/bc_rnn_th.json --name square_im \                 
    --dataset ./datasets/concatenated_square.hdf5
```

## Step 6. Evaluate improvement policy

```
    python robomimic/scripts/eval_improvement.py \
        --checkpoint diffusion_policy_trained_models/square/20250612142846/models/model_epoch_100.pth \
        --improvement_checkpoint improvement_models/tool_hang/square_im/20250612161055/models/model_epoch_200.pth
```