## INSTALL
Create the environment from the bundled YAML:

```bash
conda env create -f environment.yml
conda activate glis
```

## DOWNLOAD
Download data and models from:

https://pan.baidu.com/s/1IF77wgP8XjnToPV8s3kcgA?pwd=76fy

## TRAIN
Download `localization.pth` first. Run training from the repository root:

```bash
bash scripts/train_bin.sh
bash scripts/train_local.sh
bash scripts/train_global.sh
```

## EVALUATE

```bash
bash scripts/evaluate.sh
```
