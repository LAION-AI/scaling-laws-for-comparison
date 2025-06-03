#!/bin/bash -x
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --exclude=jwb[0056,0067-0072,0074,0093,0095-0096,0134,0158,0186,0189,0193-0202,0220,0222,0224,0232,0246,0251,0257,0265-0266,0278-0279,0285,0289-0300,0309-0320,0342,0359,0454,0459,0470,0472,0474-0492,0501-0512,0542,0549,0577-0588,0597-0604,0606,0637,0645,0647,0672,0705-0716,0725-0748,0758-0766,0829,0832,0838,0900,0907,0921,0950,0971,1004,1023,1150,1188-1191,1194-1196,1206-1217,1239-1241,1244,0812]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00
#SBATCH --partition={partition}
#SBATCH --output={output_file}
#SBATCH --job-name={full_name}
#SBATCH --open-mode=append

echo "Job Id:$SLURM_JOB_ID"

ml purge
export TRANSFORMERS_CACHE=cache
export TRANSFORMERS_OFFLINE=1
source /p/project/ccstdl/laion/mamba/bin/activate experimental-torch-nightly
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR

export SRUN_CPUS_PER_TASK=${{SLURM_CPUS_PER_TASK}}

# if siglip is true, "--siglip", else ""
if [ "{siglip}" = "True" ]; then
    SIGLIP="--siglip"
else
    SIGLIP=""
fi

if [ "{mammut}" = "True" ]; then
    MAMMUT_ARGS="--coca-contrastive-loss-weight {contrastive_weight} --coca-caption-loss-weight {caption_weight}"
else
    MAMMUT_ARGS=""
fi

if [ "{coca}" = "True" ]; then
    COCA_ARGS="--coca-contrastive-loss-weight {contrastive_weight} --coca-caption-loss-weight {caption_weight}"
else
    COCA_ARGS=""
fi

#/p/project/laionize/jitsev1_juwelsbooster/open_clip_scaling
export OPEN_CLIP_PATH=/p/project/laionize/cherti1/open_clip_scaling/open_clip

# export PYTHONPATH="$PYTHONPATH:$PWD/open_clip/src"
export PYTHONPATH="$PYTHONPATH:$OPEN_CLIP_PATH/src"

# open_clip/src/training/main.py
cd $OPEN_CLIP_PATH

srun --cpu_bind=none,v --accel-bind=gn --threads-per-core=1 python -u src/training/main.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --train-data="{train_data}"  --dataset-type webdataset --dataset-resampled\
    --train-num-samples={train_num_samples} \
    --batch-size {batch_size} $SIGLIP\
    --report-to=tensorboard \
    --epochs {epochs} \
    --workers=8 \
    --model {model}\
    --name {name} \
    --logs {logs} \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --lr {lr} \
    --beta1 {beta1} \
    --beta2 {beta2} \
    --wd {wd} $MAMMUT_ARGS $COCA_ARGS \
    --warmup {warmup} \
    --grad-clip-norm {grad_clip_norm} \
    --save-most-recent \
    --ddp-static-graph \
    --precision amp_bfloat16 \
    --grad-checkpoint \
    --resume latest
