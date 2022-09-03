# Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis
[![report](https://img.shields.io/badge/arxiv-report-red)](http://arxiv.org/abs/2204.11798) 
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() -->

![Teaser image](./docs/teaser.png)

> **Abstract:** *This work targets using a general deep learning framework to synthesize free-viewpoint images of arbitrary human performers, only requiring a sparse number of camera views as inputs and skirting per-case fine-tuning. The large variation of geometry and appearance, caused by articulated body poses, shapes, and clothing types, are the key bottlenecks of this task. To overcome these challenges, we present a simple yet powerful framework, named Generalizable Neural Performer (GNR), that learns a generalizable and robust neural body representation over various geometry and appearance. Specifically, we compress the light fields for a novel view of human rendering as conditional implicit neural radiance fields with several designs from both geometry and appearance aspects. We first introduce an Implicit Geometric Body Embedding strategy to enhance the robustness based on both parametric 3D human body model prior and multi-view source images hints. On top of this, we further propose a Screen-Space Occlusion-Aware Appearance Blending technique to preserve the high-quality appearance, through interpolating source view appearance to the radiance fields with a relaxed but approximate geometric guidance.* <br>

[Wei Cheng](mailto:wchengad@connect.ust.hk), [Su Xu](mailto:xusu@sensetime.com), [Jingtan Piao](mailto:piaojingtan@sensetime.com), [Chen Qian](https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=zh-CN), [Wayne Wu](https://wywu.github.io/), [Kwan-Yee Lin](https://kwanyeelin.github.io/), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)<br>
**[[Demo Video]](https://www.youtube.com/watch?v=2COR4u1ZIuk)** | **[[Project Page]](https://generalizable-neural-performer.github.io/)** | **[[Data]](https://generalizable-neural-performer.github.io/genebody.html)** | **[[Paper]](https://arxiv.org/pdf/2204.11798.pdf)**

## Updates
- [01/09/2022] We also recommend the implementation of our work in the [OpenXRLab](https://github.com/openxrlab/xrnerf). 
- [01/09/2022] :exclamation: GeneBody has been reframed. For users who have downloaded GeneBody before `2022.09.01` please update the latest data using our more user-friendly download tool.
- [29/07/2022] GeneBody can be downloaded from [OpenDataLab](https://opendatalab.com/GeneBody).
- [11/07/2022] Code is released.
- [02/05/2022] GeneBody Train40 is released! Apply [here](./docs/Dataset.md#train40)! 
- [29/04/2022] SMPLx fitting toolbox and benchmarks are released!
- [26/04/2022] Technical report released.
- [24/04/2022] The codebase and project page are created.

## Data Download
To download and use the GeneBody dataset set, please first read the instructions in [Dataset.md](./docs/Dataset.md). We provide a download tool to download and update the GeneBody data including dataset and pretrained models (if there is any future adjustment), for example
```
python genebody/download_tool.py --genebody_root ${GENEBODY_ROOT} --subset train40 test10 pretrained_models smpl_depth
```
The tool will fetch and download the subsets you selected and put the data in `${GENEBODY_ROOT}`.

## Annotations
GeneBody provides the per-view per-frame segmentation, using [BackgroundMatting-V2](https://github.com/PeterL1n/BackgroundMattingV2), and register the fitted [SMPLx](https://github.com/PeterL1n/BackgroundMattingV2) using our enhanced multi-view smplify repo in [here](https://github.com/generalizable-neural-performer/bodyfitting).

To use annotations of GeneBody, please check the document [Annotation.md](./docs/Annotation.md), we provide a reference data fetch module in `genebody`.

## Train and Evaluate GNR

Setup the environment 
```
conda env create -f environment.yml
conda activate gnr
pip install git+https://github.com/generalizable-neural-performer/gnr.git@mesh_grid
```

To run GNR on genebody
```
python apps/run_genebody.py --config configs/[train, test, render].txt --dataroot ${GENEBODY_ROOT}
```
if you have multiple machines and multiple GPUs, you can try to train our model using distributed data parallel
```
bash scripts/train_ddp.sh
```


## Benchmarks
We also provide benchmarks of start-of-the-art methods on GeneBody Dataset, methods and requirements are listed in [Benchmarks.md](https://github.com/generalizable-neural-performer/genebody-benchmarks).

To test the performance of our released pretrained models or train by yourselves, run:
```
git clone --recurse-submodules https://github.com/generalizable-neural-performer/gnr.git
```
And `cd benchmarks/`, the released benchmarks are ready to go on Genebody and other datasets such as V-sense and ZJU-Mocap.

### Case-specific Methods on Genebody
| Model  | PSNR | SSIM |LPIPS| ckpts|
| :--- | :----------