# Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets  [[arXiv:2506.04598]](https://arxiv.org/abs/2506.04598)

*by Marianna Nezhurina, Tomer Porian, Giovanni Puccetti, Tommie Kerssies, Romain Beaumont, Mehdi Cherti, Jenia Jitsev* [[arXiv:2506.04598]](https://arxiv.org/abs/2506.04598) [NeurIPS 2025](https://openreview.net/forum?id=cWnZLIdeKn)

In this repository, we provide detailed results and code to reproduce all figures from the paper.

## Release

### [Nov 20, 2025]

We release a total of **4,686 models** on  [HuggingFace](https://huggingface.co/laion/scaling-laws-for-comparison), consisting of all trained CLIP and MaMMUT models using cosine learning rate schedule.

**Release Highlights:**
* **Checkpoints:** Available on [HuggingFace](https://huggingface.co/laion/scaling-laws-for-comparison). Includes not only the last checkpoint of each model but also all **intermediate** checkpoints (**73,768 total**).
* **Scope:** Covers 15 model scales and 11 samples seen scales, 3 pre-training datasets (DataComp, Re-LAION, and DFN).
* **Downloads:** Use our [helper script here](#Downloading-all-model-checkpoints) to download the models.

**Evaluations:**
We also release evaluations for all models (including intermediate checkpoints of each model) on downstream tasks:
* **Zero-shot classification:** 35 datasets (DataComp evaluation suite).
* **Zero-shot retrieval:** MS-COCO.
* **Segmentation:** ADE-20K.

*See [overview.ipynb](overview.ipynb) for more details.*

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

## Downloading all model checkpoints

We are currently working on uploading all model checkpoints (including intermediate checkpoints) to HuggingFace in the following repo: <https://huggingface.co/laion/scaling-laws-for-comparison>.

We also provide a helper script ([download_models.py](download_models.py)) to download the checkpoints.

To download all models, run the following command:
```bash
python download_models.py
```

It is also possible to filter the models by using the arguments of the script. 

For example, to download only MaMMUT models trained on datacomp-1.4B dataset, with samples seen scale 1.28B, ViT-B-32 architecture, run the following command:

```bash
python download_models.py --samples_seen 1.28B --pretrain_dataset datacomp_1b --model_name ViT-B-32 --model_type mammut
```

By default, only last checkpoint of each model is downloaded, not intermediate checkpoints. To download intermediate checkpoints, use the `--download_mode all_checkpoints` argument.

See `python download_models.py -h` for more information.

## Using a downloaded model

After downloading a model, it is possible to load it in OpenCLIP directly.

For instance, if you would like to use the best ViT-B-32 MaMMUT model at 12.8B samples seen scale, run the following command:

```bash
python download_models.py --pretrain_dataset datacomp_1b --samples_seen 12.8B --model_name ViT-B-32 --model_type mammut --download_top 1
```
By default, model with best ImageNet-1k zero-shot accuracy is downloaded.

This will download the latest checkpoint at `download/mammut_ViT-B-32_s12.8B_dfn_2b_globalbs180224_lr0.0025_b1_0.9_b2_0.95_sched_cosine_warmup6000_gpus512/epoch_latest.pt`

To use the model, first, you need to install OpenCLIP MaMMUT, a fork of OpenCLIP with MaMMUT support:

```bash
git clone https://github.com/LAION-AI/open_clip_mammut
cd open_clip_mammut
python -m pip install .
```

Zero-shot classification example:

```python
import torch
from PIL import Image
import open_clip

model, _, transform = open_clip.create_model_and_transforms('mammut_ViT-B-32', pretrained='download/mammut_ViT-B-32_s12.8B_dfn_2b_globalbs180224_lr0.0025_b1_0.9_b2_0.95_sched_cosine_warmup6000_gpus512/epoch_latest.pt')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('mammut_ViT-B-32')

image = transform(Image.open("image.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.amp.autocast('cuda'):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

Caption generation example:

```python
import open_clip
import torch
from PIL import Image

model, _, transform = open_clip.create_model_and_transforms('mammut_ViT-B-32', pretrained='download/mammut_ViT-B-32_s12.8B_dfn_2b_globalbs180224_lr0.0025_b1_0.9_b2_0.95_sched_cosine_warmup6000_gpus512/epoch_latest.pt')

im = Image.open("image.png").convert("RGB")
im = transform(im).unsqueeze(0)

with torch.no_grad(), torch.amp.autocast('cuda'):
  generated = model.generate(im)

print(open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
```

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

