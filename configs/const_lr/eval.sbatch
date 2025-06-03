#!/bin/bash -x
#SBATCH --account={account}
#SBATCH --nodes={eval_nodes}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --partition={partition}
#SBATCH --output={output_file}
#SBATCH --job-name={name}
echo "Job Id:$SLURM_JOB_ID"
ml purge
source /p/project/ccstdl/laion/mamba/bin/activate experimental-torch-nightly
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python 
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --cpus-per-task=12 clip_benchmark  eval --model {model} --pretrained {logs}/{folder_name}/checkpoints/epoch_[0-9]*.pt --dataset wds/imagenet1k wds/mscoco_captions --dataset_root '/p/data1/mmlaion/vtab_plus_wds/{{dataset}}' --output '{logs}/{folder_name}/checkpoints/{{dataset}}_{{pretrained}}_{{model}}_{{language}}_{{task}}.json' --skip_existing --distributed
