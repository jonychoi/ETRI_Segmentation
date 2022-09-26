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
conda activate etri
```

#### 2-2. Fast Execute: Run bash file at the root (main.sh)
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

1. Datasets
To set the root directory of the ETRI datasets,

- Change the bash file with configurations

```
--source_root '/media/dataset2/etri' \
--target_root '/media/dataset2/DRAM_processed' \
```

or you can just

- Execute bash file with configuration

```
bash main.sh --source_root '/media/dataset2/etri'
```

### 4. Model and Log Saving Scheme

#### 4-1. Weights saving
Set args.save_dir at the main.sh
```
--save_dir "/media/dataset2/etri_result/etri_semi/"
```

Best model will be saved as

```
save_dir
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