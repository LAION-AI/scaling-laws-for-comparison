# Eval Plus

create extended evals for already pre-trained models.

## Step 1: create yaml config

Example:

`python make_config.py  --folder /p/data1/mmlaion/experiments/autoexp/jjitsev/logs --tasks zeroshot_classification image_caption_selection losses`

## Step 2: Build sbatch scripts

`autoexperiment build config.yaml`

## Step 3: Run evals

`autoexperiment run config.yaml`
