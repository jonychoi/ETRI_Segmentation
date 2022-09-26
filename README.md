# ETRI_segmentation
### Semantic Segmentation for Arts - ETRI Project
This repository is for private usage. 2022 Computer Vision Lab @ Korea University.

---

### 1. Project Configuration

```
etri
├── backbone: DPT-Hybrid Configuration customized for ETRI data.
│   └── dpt: dpt backbone scripts
├── datasets: DRAM / ETRI datasets configurations
│   └── dram
├── perturbations: photometric / geometric augmentations for consistency learning
├── style_transfer: style transfer network
├── utils: label mapper, evaluator, ema model, pallete, arguments parsers, tensorboard logger etc.
├──.gitignore
├── main.py
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

> 1. At the main.sh bash file, change the configurations

> 2. Or you can just execute bash file with configurations.

E.g) if you want to execute main.py with the pseudolabel-threshold of 0.8,
```
bash main.sh --pseudolabel_threshold 0.8
```




### 3. Additional Weights and Dataset Configuration

1. ETRI Datasets

2. DRAM Datasets


### 4. Model Saving Scheme      