# Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets  [[arXiv]](https://arxiv.org/abs/)

*by Marianna Nezhurina, Tomer Porian, Giovanni Pucceti, Tommie Kerssies, Romain Beaumont, Mehdi Cherti, Jenia Jitsev* [[arXiv:]](https://arxiv.org/abs/)

In this repository, we provide detailed results and code to reproduce all figures from the paper.

## Main Results
We demonstrate scaling law derivation based model and dataset comparison. As working example, we compare contrastive loss based CLIP and contrastive + text generative (captioning) loss based MaMMUT, using open datasets Re-LAION-1.4B, DataComp-1.4B and DFN-1.4B. In plots below, we illustrate consistent stronger scalability of MaMMUT across datasets and downstream tasks, as well as stronger performance when training on DFN-1.4B for both CLIP and MaMMUT.  


## Detailed Results

In [overview.ipynb](overview.ipynb), you can view detailed results of all models that we
trained.

## Fit scaling laws to the data

To reproduce all figures from the paper, you can use the provided bash scripts. These scripts, located in the `scripts` directory, are responsible for generating each figure presented in the paper by running the necessary data processing and plotting commands.

To execute all the scripts in the correct order and reproduce the figures from the paper, use the [reproduce_all_figures.sh](reproduce_all_figures.sh) script. This master script runs each figure script (e.g., fig1.sh, etc.) sequentially:
```bash
bash reproduce_all_figures.sh
```

The master script will run each figure script and generate the corresponding plots in the `scaling_laws/derived_scaling_laws`directory.

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{nezhurina2025scaling,
  title={Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets},
  author={Marianna Nezhurina, Tomer Porian, Giovanni Pucceti, Tommie Kerssies, Romain Beaumont, Mehdi Cherti, Jenia Jitsev},
  journal={arXiv},
  year={2025}
}```
