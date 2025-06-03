import os
from glob import glob
import yaml
import pandas as pd
import argparse
from yaml import CLoader as Loader, CDumper as Dumper

def get_params(out_file):
    dic = {}
    for l in open(out_file).readlines():
        index = l.find("| INFO |")
        if index >= 0 and ":" in l:
            l = l[index+len("| INFO |"):]
            l = l.strip()
            try:
                k, v = l.split(":")
                k = k.strip()
                v = v.strip()
                dic[k] = v
            except Exception:
                pass
    return dic
            
def main(args):
    dic = yaml.load(open("config_base.yaml").read(), Loader)
    datasets = [
        l.strip().split(" ")
        for l in open(args.dataset).readlines()
    ]
    datasets = pd.read_csv(args.dataset, sep=" ")
    dic['experiments'] = []
    for folder in args.folder:
        for name in os.listdir(folder):
            folder_name = os.path.abspath(os.path.join(folder, name))
            out_file = os.path.join(folder_name, "out.log")
            if not os.path.exists(out_file):
                continue
            params = get_params(out_file)
            model = params["model"]
            for index, row in datasets.iterrows():
                suite_name = row.suite_name
                tasks = row.tasks
                ds = row.datasets
                nodes = row.nodes
                distributed = row.distributed
                root = row.dataset_root
                for task in tasks.split(","):
                    if args.tasks and task not in args.tasks:
                        continue
                    if model.startswith("mammut_"):
                        cw = float(params["coca_contrastive_loss_weight"])
                        ns = "cap" if cw == 0 else "mammut"
                    elif model.startswith("coca_"):
                        ns = "coca"
                    else:
                        ns = "clip"
                    can_cap = ns in ("coca", "cap", "mammut")
                    if "generative" in task and not can_cap:
                        # task need log-likelihood but model can't compute it
                        continue
                    if "generative" not in task and ns == "cap":
                        # task need embeddings but model can't compute it
                        continue
                    dist = {
                        "checkpoints": "--distributed",
                        "data": "--distributed-data-parallel",
                        "none": "",
                    }[distributed]
                    epochs = params["epochs"]
                    full_name = f"{name}_{task}_{suite_name}"
                    dic['experiments'].append( {full_name: {
                        "full_name": full_name,
                        "suite_name": suite_name,
                        "model": model,
                        "nodes": nodes,
                        "folder_name": folder_name,
                        "dataset": " ".join(ds.split(",")),
                        "nb_datasets": len(ds.split(",")),
                        "task": task,
                        "distributed": dist,
                        "prompt_batch_size": args.prompt_batch_size,
                        "batch_size": args.batch_size,
                        "language": "en",
                        "epochs": epochs,
                        "dataset_root": root,
                    }})
    with open(args.out, "w") as fd:
        fd.write(yaml.dump(dic))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', nargs="+", type=str)
    parser.add_argument('--out', default="config.yaml", type=str)
    parser.add_argument('--dataset', default="datasets.txt", type=str)
    parser.add_argument('--prompt_batch_size', default=32, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--tasks', default=None, nargs="+", type=str)
    args = parser.parse_args()

    main(args)
