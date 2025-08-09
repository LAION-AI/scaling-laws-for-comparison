# Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets  [[arXiv:2506.04598]](https://arxiv.org/abs/2506.04598)

*by Marianna Nezhurina, Tomer Porian, Giovanni Pucceti, Tommie Kerssies, Romain Beaumont, Mehdi Cherti, Jenia Jitsev* [[arXiv:2506.04598]](https://arxiv.org/abs/2506.04598)

In this repository, we provide detailed results and code to reproduce all figures from the paper.

## Main Results
We demonstrate scaling law derivation based model and dataset comparison. As working example, we compare contrastive loss based CLIP and contrastive + text generative (captioning) loss based MaMMUT, using open datasets Re-LAION-1.4B, DataComp-1.4B and DFN-1.4B. In plots below, we illustrate consistent stronger scalability of MaMMUT across datasets and downstream tasks (zero-shot evaluation), as well as stronger performance when training on DFN-1.4B for both CLIP and MaMMUT.  

### Model comparison: openCLIP and openMaMMUT
![image](https://github.com/user-attachments/assets/869ae40c-7f1b-4f99-928a-c41b38d90da3)

![image](https://github.com/user-attachments/assets/6d1f3881-ad93-4623-87ad-358c20e383f0)

![image](https://github.com/user-attachments/assets/25eb62a0-772f-4443-87be-76a143a16334)


### Dataset comparison: Re-LAION-1.4B, DataComp-1.4B and DFN-1.4B.

![image](https://github.com/user-attachments/assets/a7139891-991c-46d0-ad48-cea8d76c38b2)

![image](https://github.com/user-attachments/assets/9eb27cd7-3790-4ab1-b5dd-9600319fb634)

### Predictions from derived scaling laws for CLIP and MaMMUT
Established scaling laws allow us to make accurate predictions, here on example of important L-14 12.8B scales, extrapolating about 4x beyond compute scale used for scaling law measurements. 

![image](https://github.com/user-attachments/assets/79b09cce-3cba-4078-b05a-efaf3ef205b6)


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

## OpenMammut-L-14 12.8B DataComp-1.4B

OpenMaMMUT-L-14 12.8B is a large scale open vision-language foundation model, trained using insights from scaling analysis to guide choice of model and samples seen scale.
OpenMaMMUT-L-14 12.8B achieves **state of the art performance** on zero-shot classification and retrieval tasks among similar-sized models trained only on publicly-available data (MetaCLIP, DataComp, OpenVision).  It outperforms with $80.34$\% IN1k zero-shot accuracy - as predicted from scaling laws - openCLIP pre-trained on same DataComp-1.4B budget of 12.8B samples seen ($79.2$\%) and even rivals models with much larger pre-training compute like SigLIP. OpenMaMMUT represents a highly performant, fully reproducible alternative to other models, being based on **openly available data and training source code**. Note that on 12.8B samples seen scale the performance suffers from high amount of repetitions, and therefore is below our prediction of $82.0$\% (see Table 1 above and [[arXiv:2506.04598]](https://arxiv.org/abs/2506.04598) for more details) that is valid for training on unique samples.

<img src="https://cdn-uploads.huggingface.co/production/uploads/6355b485b8b79340d4630dd5/mCNQu13oNcdHasaNo3lST.png" alt="openmammut_release_logo" width="60%"/>

HuggingFace Model Repo, with examples of model usage: [OpenMaMMUT-L-14 12.8B DataComp-1.4B](https://huggingface.co/laion/openMaMMUT-ViT-L-14-DataComp-1.4B-s12.8B-b180K)

[openMammut-L-14 12.8B DataComp-1.4B](https://huggingface.co/laion/openMaMMUT-ViT-L-14-DataComp-1.4B-s12.8B-b180K) was trained using code from [custom openCLIP+Mammut fork](https://github.com/LAION-AI/open_clip_mammut) and automated experiments workflow [autoexperiment](https://github.com/SLAMPAI/autoexperiment)


### OpenMammut-L-14 12.8B performance in comparison to other reference models

<img src="https://cdn-uploads.huggingface.co/production/uploads/6355b485b8b79340d4630dd5/bLHbtJ66mxs6ErKaqbXe9.png" alt="openmammut_hyperparams" width="80%"/>

### Training hyperparameters
<img src="https://cdn-uploads.huggingface.co/production/uploads/6355b485b8b79340d4630dd5/3m_kj2FTOcOkuucb1qeFd.png" alt="openmammut_hyperparams" width="40%"/>

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{nezhurina2025scaling,
  title={Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets},
  author={Nezhurina, Marianna and Porian, Tomer and Pucceti, Giovanni and Kerssies, Tommie and Beaumont, Romain and Cherti, Mehdi and Jitsev, Jenia},
  journal={arXiv:2506.04598},
  url={https://arxiv.org/abs/2506.04598},
  year={2025}
}
```

## Acknowledgements

Authors acknowledge funding by the Federal Ministry of Education and Research of Germany (BMBF) under grant no. 01IS24085C (OPENHAFM), under the grant 16HPC117K (MINERVA) and under the grant no. 01IS22094B (WestAI - AI Service Center West), as well as co-funding by EU from EuroHPC Joint Undertaking programm under grant no. 101182737 (MINERVA) and from Digital Europe Programme under grant no. 101195233 (openEuroLLM).

Authors acknowledge the Gauss Centre for Supercomputing e.V. for funding this work by providing computing time through the John von Neumann Institute for Computing (NIC) on the supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC), EuroHPC Joint Undertaking for computing time and storage on the EuroHPC supercomputer LEONARDO, hosted by CINECA (Italy) and the LEONARDO consortium through an EuroHPC Extreme Access grant EHPC-EXT-2023E02-068, storage resources on JUST granted and operated by JSC and supported by Helmholtz Data Federation (HDF), computing time granted by the JARA and JSC on the supercomputer JURECA at JSC, and computing time granted on prototype JEDI via JUREAP (JUPITER Early Access Programm) grant at JSC. 

Further thanks go for support provided by supercomputing facilities and their teams, especially to Damian Alvarez and Mathis Bode from Juelich Supercomputer Center (JSC, Germany) and to Laura Morselli from CINECA (Italy).

Authors also would like to express gratitude to all the people who are working on making code, models and data publicly available, advancing community based research and making research more reproducible. Specifically, we would like to thank all the members of the [LAION Discord server](https://discord.gg/BZqhreFazY) community and [Open-$`\Psi`$ (Open-Sci) Collective](https://discord.gg/GsKh4mBVcv) for providing fruitful ground for scientific exchange and open-source development. 

