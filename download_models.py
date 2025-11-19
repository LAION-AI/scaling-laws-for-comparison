import os
import argparse
import shutil
import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "laion/scaling-laws-for-comparison"
repo_type = "model"

SAMPLES_SEEN = [     
    "1.28M",
    "3M",
    "6.4M",
    "12.8M",
    "30M",
    "64M",
    "128M",
    "300M",
    "640M",
    "1.28B",
    "3B",
    "6.4B",
    "12.8B",
]

PRETRAIN_DATASETS = [
    "datacomp_1b",
    "dfn_2b",
    "relaion2b-en",
]

MODEL_NAMES = [
    'ViT-M-32', 'ViT-L-14', 'ViT-S-32', 'ViT-B-32', 'ViT-S-16',
    'ViT-H-14', 'ViT-B-16', 'ViT-M-14', 'ViT-M-16',
    'ViT-B-14', 'ViT-S-14', 'ViT-L-32', 'ViT-L-16', 
    'ViT-H-16', 'ViT-H-32',
]

MODEL_TYPES = [
    "clip",
    "mammut",
    "coca",
    "cap",
    "siglip",
]

LR_SCHEDULERS = [
    "cosine",
    "const",
    "const-cooldown",
]

def download_checkpoints(folder, files, output_folder):
    for filename in files:
        hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=os.path.join(folder, filename), local_dir=output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to download pre-trained models")
    parser.add_argument("--samples_seen", type=str, nargs="+", default=[], help="Samples seen", choices=SAMPLES_SEEN)
    parser.add_argument("--pretrain_dataset", type=str, nargs="+",default=[], help="Pretrain dataset", choices=PRETRAIN_DATASETS)
    parser.add_argument("--model_name", type=str, nargs="+", default=[], help="Model name", choices=MODEL_NAMES)
    parser.add_argument("--model_type", type=str, nargs="+", default=[], help="Model type", choices=MODEL_TYPES)
    parser.add_argument("--lr_scheduler", type=str, nargs="+", default=[], help="LR scheduler", choices=LR_SCHEDULERS)
    parser.add_argument("--download_mode", type=str, default="last_checkpoint", choices=["last_checkpoint", "all_checkpoints"], help="Download mode. `last_checkpoint downloads only the last checkpoint, all_checkpoints downloads all checkpoints")
    parser.add_argument("--download_top", type=int, default=-1, help="Download top-k models according to imagenet1k zero-shot accuracy. By default, download all models")
    parser.add_argument("--output_folder", type=str, default="download", help="Output folder")
    args = parser.parse_args()

    cache_dir = os.path.join(args.output_folder, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    results_filepath = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename="results.parquet",
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    df = pd.read_parquet(results_filepath)

    df = df[df.epoch == df.total_epochs]

    if args.pretrain_dataset:
        df = df[df.pretrain_dataset.isin(args.pretrain_dataset)]
    if args.model_name:
        df = df[df.model_simple.isin(args.model_name)]
    if args.model_type:
        df = df[df.namespace.isin(args.model_type)]
    if args.lr_scheduler:
        df = df[df.lr_scheduler.isin(args.lr_scheduler)]
    if args.samples_seen:
        df = df[df.samples_seen_scale_pretty.isin(args.samples_seen)]
    
    # Sort models by imagenet1k zero-shot accuracy
    df = df[df.downstream_dataset=="imagenet1k"]
    df = df.sort_values(by='acc1', ascending=False)

    # Once filtered, download only top-k models according imagenet1k zero-shot accuracy
    # with k = args.download_top
    # By default, download all models
    if args.download_top > 0:
        df = df.iloc[:args.download_top]

    for _, row in df.iterrows():
        if args.download_mode == "last_checkpoint":
            files = [f"epoch_{row.total_epochs}.pt"]
        elif args.download_mode == "all_checkpoints":
            files = [f"epoch_{epoch}.pt" for epoch in range(1, row.total_epochs+1)]
        else:
            raise ValueError("Invalid download mode")

        print(f"Downloading {row.full_name} with ImageNet-1k zero-shot accuracy {row.acc1:.2f}")
        download_checkpoints(folder=row.full_name, files=files, output_folder=args.output_folder)
        os.symlink(
            os.path.abspath(os.path.join(args.output_folder, row.full_name, f"epoch_{row.total_epochs}.pt")), 
            os.path.abspath(os.path.join(args.output_folder, row.full_name, "epoch_latest.pt"))
        )
