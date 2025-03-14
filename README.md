# PRIME: Scaffolding Manipulation Tasks with Behavior Primitives for Data-Efficient Imitation Learning
<br>

This is the codebase for the [**PRIME**](https://ut-austin-rpl.github.io/prime/) paper:

**PRIME: Scaffolding Manipulation Tasks with Behavior Primitives for Data-Efficient Imitation Learning**
<br> [Tian Gao](https://skybhh19.github.io/), [Soroush Nasiriany](http://snasiriany.me/), [Huihan Liu](https://https://huihanl.github.io/), [Quantao Yang](https://yquantao.github.io/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/) 
<br> [UT Austin Robot Perception and Learning Lab](https://rpl.cs.utexas.edu/)
<br> IEEE Robotics and Automation Letters (RA-L), 2024
<br> **[[Paper]](http://arxiv.org/abs/2403.00929)** &nbsp;**[[Project Website]](https://ut-austin-rpl.github.io/PRIME/)** 

<a href="https://ut-austin-rpl.github.io/prime/" target="_blank"><img src="images/pull_figure.png" width="90%" /></a>

<br>

## Installation
```commandline
git clone https://github.com/UT-Austin-RPL/PRIME.git prime

# For cuda 12
conda create -n prime python=3.8
conda activate prime
pip install torch torchvision torchaudio  

# For cuda 11
conda create -n prime python=3.7.9
conda activate prime
pip install torch==1.13.1 torchvision==0.14.1
```
Install robosuite.
```commandline
git clone -b prime https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
pip install -r requirements.txt
pip install -r requirements-extra.txt
# For cuda 11
pip install mujoco-py==2.1.2.14 "Cython<3"
```
Install robomimic.
```commandline
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
git checkout b5d2aa9902825c6c652e
pip install -e .
```
Install requirements.
```commandline
cd prime
pip install -r requirements.txt     
```

## Data collection

### Human demonstration collection

We collect 30 human demonstrations for each task using [Spacemouse](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/using_teleoperation_devices.html). 
```commandline
cd robosuite
python robosuite/scripts/collect_human_demonstrations.py --environment CleanUpMediumSmallInitD2 --directory ../prime/data/human_demos --only-yaw --only-success --device spacemouse
python robosuite/scripts/collect_human_demonstrations.py --environment NutAssemblyRoundSmallInit --directory ../prime/data/human_demos --only-yaw --only-success --device spacemouse
python robosuite/scripts/collect_human_demonstrations.py --environment PickPlaceMilk --directory ../prime/data/human_demos --only-yaw --only-success --device spacemouse
```
Convert the demonstrations to the format that can be used for training.
```commandline
cd prime

# TidyUp
python scripts/conversion/convert_robosuite.py --dataset ../prime/data/human_demos/CleanUpMediumSmallInitD2/demo.hdf5
python scripts/dataset_states_to_obs.py --dataset ../prime/data/human_demos/CleanUpMediumSmallInitD2/demo.hdf5 --output_name demo_robomimic.hdf5 --camera_names agentview robot0_eye_in_hand
# NutAssembly
python scripts/conversion/convert_robosuite.py --dataset ../prime/data/human_demos/NutAssemblyRoundSmallInit/demo.hdf5
python scripts/dataset_states_to_obs.py --dataset ../prime/data/human_demos/NutAssemblyRoundSmallInit/demo.hdf5 --output_name demo_robomimic.hdf5 --camera_names agentview robot0_eye_in_hand
# PickPlace
python scripts/conversion/convert_robosuite.py --dataset ../prime/data/human_demos/PickPlaceMilk/demo.hdf5
python scripts/dataset_states_to_obs.py --dataset ../prime/data/human_demos/PickPlaceMilk/demo.hdf5 --output_name demo_robomimic.hdf5 --camera_names agentview robot0_eye_in_hand
```
### Data collection for Inverse Dynamics Models (IDMs)
```commandline
cd prime
python scripts/data_processing.py --collect-demos --reformat-rollout-data --data-dir data/rollout_data --env CleanUpMediumSmallInitD2 --num-trajs 40 --num-primitives 50 --save --num-data-workers 70 --num-others-per-traj 100 --policy-pretrain --verbose  # TidyUp
python scripts/data_processing.py --collect-demos --reformat-rollout-data --data-dir data/rollout_data --env NutAssemblyRoundSmallInit --num-trajs 120 --num-primitives 15 --save --num-data-workers 45 --num-others-per-traj 60 --policy-pretrain --verbose  # NutAssembly
python scripts/data_processing.py --collect-demos --reformat-rollout-data --data-dir data/rollout_data --env PickPlaceMilk --num-trajs 50 --num-primitives 15 --save --num-data-workers 45 --num-others-per-traj 60 --policy-pretrain --verbose  # PickPlace
```

## Training Trajectory Parser

### Training IDMs

```commandline
cd prime

# TidyUp
python scripts/train_robomimic_models.py --config prime/exps/CleanUpMediumSmallInitD2/idm/type/seed1.json 
python scripts/train_robomimic_models.py --config prime/exps/CleanUpMediumSmallInitD2/idm/params/seed1.json 
# NutAssembly
python scripts/train_robomimic_models.py --config prime/exps/NutAssemblyRoundSmallInit/idm/type/seed1.json 
python scripts/train_robomimic_models.py --config prime/exps/NutAssemblyRoundSmallInit/idm/params/seed1.json 
# PickPlace
python scripts/train_robomimic_models.py --config prime/exps/PickPlaceMilk/idm/type/seed1.json 
python scripts/train_robomimic_models.py --config prime/exps/PickPlaceMilk/idm/params/seed1.json 
```

### Segmenting human demonstrations
```commandline
cd prime
python scripts/data_processing.py --segment-demos --demo-path data/human_demos/CleanUpMediumSmallInitD2/demo_robomimic.hdf5 --idm-type-model-path=trained_models/CleanUpMediumSmallInitD2/idm_ckpts/type/idm_type_seed1/20250202021653/models/model_epoch_300.pth --idm-params-model-path=trained_models/CleanUpMediumSmallInitD2/idm_ckpts/params/idm_params_seed1/20250202021704/models/model_epoch_600.pth --segmented-data-dir data/human_demos/CleanUpMediumSmallInitD2/segmented_data/seed1 --save-failed-trajs --max-primitive-horizon=80 --playback-segmented-trajs --verbose --num-augmentation-type 50 --num-augmentation-params 100  # TidyUp
python scripts/data_processing.py --segment-demos --demo-path data/NutAssemblyRoundSmallInit/demo_robomimic.hdf5 --idm-type-model-path=trained_models/NutAssemblyRoundSmallInit/idm_ckpts/type/idm_type_seed1/20250120150133/models/model_epoch_300.pth --idm-params-model-path=trained_models/NutAssemblyRoundSmallInit/idm_ckpts/params/idm_params_seed1/20250120151327/models/model_epoch_600.pth --segmented-data-dir data/human_demos/NutAssemblyRoundSmallInit/segmented_data/seed1 --save-failed-trajs --max-primitive-horizon=200 --playback-segmented-trajs --verbose --num-augmentation-type 50 --num-augmentation-params 100  # NutAssembly
python scripts/data_processing.py --segment-demos --demo-path data/PickPlaceMilk/demo_robomimic.hdf5 --idm-type-model-path=trained_models/PickPlaceMilk/idm_ckpts/type/idm_type_seed1/20250121052558/models/model_epoch_300.pth --idm-params-model-path=trained_models/PickPlaceMilk/idm_ckpts/params/idm_parmas_seed1/20250122040510/models/model_epoch_600.pth --segmented-data-dir data/human_demos/PickPlaceMilk/segmented_data/seed1 --save-failed-trajs --max-primitive-horizon=200 --playback-segmented-trajs --verbose --num-augmentation-type 50 --num-augmentation-params 100  # PickPlace
```

## Training Policy with primitives

### Pre-training policy
```commandline
cd prime
python scripts/train_robomimic_models.py --config prime/exps/CleanUpMediumSmallInitD2/policy/pt/params/seed1.json  # TidyUP
python scripts/train_robomimic_models.py --config prime/exps/NutAssemblyRoundSmallInit/policy/pt/params/seed1.json  # NutAssembly
python scripts/train_robomimic_models.py --config prime/exps/PickPlaceMilk/policy/pt/params/seed1.json  # PickPlace
```

### Fine-tuning policy
```commandline
cd prime

# TidyUp
python scripts/train_robomimic_models.py --config prime/exps/CleanUpMediumSmallInitD2/policy/ft/type/seed1.json 
python scripts/train_robomimic_models.py --config prime/exps/CleanUpMediumSmallInitD2/policy/ft/params/seed1.json 
# NutAssembly
python scripts/train_robomimic_models.py --config prime/exps/NutAssemblyRoundSmallInit/policy/ft/type/seed1.json 
python scripts/train_robomimic_models.py --config prime/exps/NutAssemblyRoundSmallInit/policy/ft/params/seed1.json 
# PickPlace
python scripts/train_robomimic_models.py --config prime/exps/PickPlaceMilk/policy/ft/type/seed1.json 
python scripts/train_robomimic_models.py --config prime/exps/PickPlaceMilk/policy/ft/params/seed1.json 
``` 

### Policy evaluation
```commandline
cd prime
python scripts/eval_policy.py --policy-type-model-dir trained_models/CleanUpMediumSmallInitD2/policy_ft_ckpts/type/policy_ft_type_seed1/20250202043934/models/ --policy-params-model-dir trained_models/CleanUpMediumSmallInitD2/policy_ft_ckpts/params/policy_ft_params_seed1/20250202043033/models/ --env-horizon 800  # TidyUp
python scripts/eval_policy.py --policy-type-model-dir trained_models/NutAssemblyRoundSmallInit/policy_ft_ckpts/type/policy_ft_type_seed1/20250202042844/models/ --policy-params-model-dir trained_models/NutAssemblyRoundSmallInit/policy_ft_ckpts/params/policy_ft_params_seed2/20250202040709/models/ --env-horizon 400  # NutAssembly
python scripts/eval_policy.py --policy-type-model-dir trained_models/PickPlaceMilk/policy_ft_ckpts/type/policy_ft_type_seed1/20250202042905/models/ --policy-params-model-dir trained_models/PickPlaceMilk/policy_ft_ckpts/params/policy_ft_params_seed1/20250202031947/models/ --env-horizon 400  # PickPlace
```

## Citation
```bibtex
@article{gao2024prime,
  title={PRIME: Scaffolding Manipulation Tasks with Behavior Primitives for Data-Efficient Imitation Learning},
  author={Tian Gao and Soroush Nasiriany and Huihan Liu and Quantao Yang and Yuke Zhu},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2024}
}
```

