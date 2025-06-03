#!/bin/bash -x
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --output={output_file}
#SBATCH --job-name={name}
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
srun --cpu_bind=v --cpus-per-task=12 clip_benchmark  eval --model {model} --pretrained {folder_name}/checkpoints/epoch_[0-9]*.pt --dataset {dataset} --dataset_root '{dataset_root}/{{dataset}}' --output '{folder_name}/checkpoints/eval_plus_{suite_name}_{{dataset}}_{{pretrained}}_{{model}}_{{language}}_{{task}}.json' --skip_existing  --task {task}  --batch_size {batch_size} --prompt_batch_size {prompt_batch_size} {distributed}
