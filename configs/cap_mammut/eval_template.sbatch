#!/bin/bash -x
#SBATCH --account={account}
#SBATCH --nodes={eval_nodes}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time={eval_time}
#SBATCH --partition={partition}
#SBATCH --output={output_file}
#SBATCH --job-name={name}_{mode}
echo "Job Id:$SLURM_JOB_ID"
export TRANSFORMERS_CACHE=cache
export TRANSFORMERS_OFFLINE=1
source /p/project/ccstdl/laion/mamba/bin/activate experimental-torch-nightly
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR
export OPEN_CLIP_PATH=/p/project/laionize/cherti1/open_clip_scaling/open_clip
export PYTHONPATH="$PYTHONPATH:$OPEN_CLIP_PATH/src"

srun --cpu_bind=v --cpus-per-task=12 clip_benchmark  eval --model {model} --pretrained {logs}/{folder_name}/checkpoints/epoch_[0-9]*.pt --dataset winoground sugar_crepe --dataset_root '/p/data1/mmlaion/vtab_plus_wds/{{dataset}}' --output '{logs}/{folder_name}/checkpoints/{{dataset}}_{{pretrained}}_{{model}}_{{language}}_{{task}}.json' --skip_existing --distributed --task generative_image_caption_selection

srun --cpu_bind=v --cpus-per-task=12 clip_benchmark  eval --model {model} --pretrained {logs}/{folder_name}/checkpoints/epoch_[0-9]*.pt --dataset imagenet1k --dataset_root '/p/data1/mmlaion/vtab_plus/imagenet1k_5kval' --output '{logs}/{folder_name}/checkpoints/{{dataset}}_{{pretrained}}_{{model}}_{{language}}_{{task}}.json' --skip_existing --distributed-data-parallel --task generative_zeroshot_classification --batch_size 64 --prompt_batch_size 32
