# ETRI_segmentation
### Semantic Segmentation for Arts - ETRI Project
This repository is for private usage. 2022 Computer Vision Lab @ Korea University.

---

### 1. Project Configuration

```
etri
├── backbone: DPT-Hybrid Configuration customized for ETRI data.
│   └── dpt:: dpt backbone scripts
├── datasets:: DRAM / ETRI datasets configurations
│   └── dram
├── perturbations:: photometric / geometric augmentations for consistency learning
├── style_transfer:: style transfer network
├── utils:: label mapper, evaluator, ema model, pallete, arguments parsers, tensorboard logger etc.
├──.gitignore
├── main.py:: Semi Supervised Consistency Learning
├── main.sh
├── README.md
```


### 2. Code of Conduct
All instructions are instructed at the root directory of the project.

#### 2-1. Activate the Conda Environment
```
conda env create -f environment.yaml
conda activate segmento
```

#### 2-1. Inference: Run bash file at the root (test.sh)
```
bash test.sh
```

#### 2-2. Training: Run bash file at the root (main.sh)
```
bash main.sh
```

#### 2-3. Tunning the Configurations

- Changing Configurations at the main.sh

```
...
--target_dataset 'DRAM' \
--lam_style 1.0 \
--pseudolabel_threshold 0.8 \
--lam_randaug 0.8 \
--lam_styleaug 0.8 \
...
```

or you can just

- Execute bash file with configurations

E.g) if you want to execute main.py with the pseudolabel-threshold of 0.8,
```
bash main.sh --pseudolabel_threshold 0.8
```


### 3. Additional Weights and Dataset Configuration

3-1. Datasets
To set the root directory of the ETRI datasets,

- Change the bash file with configurations

```
--source_root './dataset/etri' \
--target_root './dataset/DRAM_processed' \
```

or you can just

- Execute bash file with configuration

```
bash main.sh --source_root './dataset/etri'
```

3-2. Pretrained Weights

To start from the ade20k pretrained weights of DPT-Hybrid, you should set the directory of the weights.

- Change the bash file with configurations

```
--pretrained "./dpt_hybrid-ade20k-53898607.pt" \
```

### 4. Model and Log Saving Scheme

#### 4-1. Weights saving
Set args.save_path at the main.sh
```
--save_path "./checkpoints"\
```

Best model will be saved as

```
save_path
├── experiment_name
│   └── source
│       └── top_K
│           └── epoch::{EPOCH}::iter::{ITERS}::model::dpt_hybrid::miou::{MIOU}.pth
```

#### 4-2. Tensorboard Logging
Tensorboard logs are saved at
```
root/tensorboard/{EXPERIMENT_NAME}/{DATE}/
```
To execute the tensorboard, write the command at the terminal as following.
```
tensorboard dev upload --logdir {TENSORBOARD LOG DIR} --name {ANY NAME}
```

#### 5. Default Classes

1. Person
2. Sky+Cloud
3. Grass + Ground-others
4. Wall-others
5. Tree
6. Water-others


MIoUs will be updated soon.
