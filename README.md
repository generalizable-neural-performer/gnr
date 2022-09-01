# Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis
[![report](https://img.shields.io/badge/arxiv-report-red)](http://arxiv.org/abs/2204.11798) 
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() -->

![Teaser image](./docs/teaser.png)

> **Abstract:** *This work targets at using a general deep learning framework to synthesize free-viewpoint images of arbitrary human performers, only requiring a sparse number of camera views as inputs and skirting per-case fine-tuning. The large variation of geometry and appearance, caused by articulated body poses, shapes and clothing types, are the key bot tlenecks of this task. To overcome these challenges, we present a simple yet powerful framework, named Generalizable Neural Performer (GNR), that learns a generalizable and robust neural body representation over various geometry and appearance. Specifically, we compress the light fields for novel view human rendering as conditional implicit neural radiance fields with several designs from both geometry and appearance aspects. We first introduce an Implicit Geometric Body Embedding strategy to enhance the robustness based on both parametric 3D human body model prior and multi-view source images hints. On the top of this, we further propose a Screen-Space Occlusion-Aware Appearance Blending technique to preserve the high-quality appearance, through interpolating source view appearance to the radiance fields with a relax but approximate geometric guidance.* <br>

[Wei Cheng](mailto:wchengad@connect.ust.hk), [Su Xu](mailto:xusu@sensetime.com), [Jingtan Piao](mailto:piaojingtan@sensetime.com), [Chen Qian](https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=zh-CN), [Wayne Wu](https://wywu.github.io/), [Kwan-Yee Lin](https://kwanyeelin.github.io/), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)<br>
**[[Demo Video]](https://www.youtube.com/watch?v=2COR4u1ZIuk)** | **[[Project Page]](https://generalizable-neural-performer.github.io/)** | **[[Data]](https://generalizable-neural-performer.github.io/genebody.html)** | **[[Paper]](https://arxiv.org/pdf/2204.11798.pdf)**

## Updates
- [01/09/2022] GNR has been reimplemented in the framework of [openxrlab/xrnerf](https://github.com/openxrlab/xrnerf). 
- [01/09/2022] :exclamation: GeneBody has been reframed. For users have downloaded GeneBody before `2022.09.01` please update the latest data using our more user-friendly download tool.
- [29/07/2022] GeneBody can be downloaded from [OpenDataLab](https://opendatalab.com/GeneBody).
- [11/07/2022] Code is released.
- [02/05/2022] GeneBody Train40 is released! Apply [here](./docs/Dataset.md#train40)! 
- [29/04/2022] SMPLx fitting toolbox and benchmarks are released!
- [26/04/2022] Techincal report released.
- [24/04/2022] The codebase and project page are created.

## Data Download
To download and use the GeneBody dataset set, please frist read the instructions in [Dataset.md](./docs/Dataset.md). We provide a download tool to download and update the GeneBody data including dataset and pretrained models (if there is any future adjustment), for example
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
cd lib/mesh_grid && python setup.py install
```

To run GNR on genebody
```
python apps/run_genebody.py --config configs/[train, test, render].txt --dataroot ${GENEBODY_ROOT}
```
if you have multiple machines and mulitple GPUs, you can try to train our model using distributed data parrallel
```
bash scripts/train_ddp.sh
```


## Benchmarks
We also provide benchmarks of start-of-the-art methods on GeneBody Dataset, methods and requirements are listed in [Benchmarks.md](https://github.com/generalizable-neural-performer/genebody-benchmarks).

To test the performance of our released pretrained models, or train by yourselves, run:
```
git clone --recurse-submodules https://github.com/generalizable-neural-performer/gnr.git
```
And `cd benchmarks/`, the released benchmarks are ready to go on Genebody and other datasets such as V-sense and ZJU-Mocap.

### Case-specific Methods on Genebody
| Model  | PSNR | SSIM |LPIPS| ckpts|
| :--- | :---------------:|:---------------:| :---------------:| :---------------:  |
| [NV](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nv)| 19.86 |0.774 |  0.267 | [ckpts](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/EniK9r9UdbtGvYvtJITBGkIBlmxSHqaoEIiIgpYBGddCHQ?e=RbS0sG)|
| [NHR](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nhr)| 20.05  |0.800 |  0.155 | [ckpts](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/EqQDNVch2j5DmyIDnHX0VgkBDdCksmT4Kfq2oPOMn6gfMg?e=dy6yUA)|
| [NT](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nt)| 21.68  |0.881 |   0.152 | [ckpts](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/Etg3LW44m61OjZOgDp-f4TcB_rgm_32ve529z5EZgCmoGw?e=zGUadc)|
| [NB](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nb)| 20.73  |0.878 |  0.231 | [ckpts](https://hkustconnect-my.sharepoint.com/personal/wchengad_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fwchengad%5Fconnect%5Fust%5Fhk%2FDocuments%2Fgenebody%2Dbenchmark%2Dpretrained%2Fnb%2Fgenebody)|
| [A-Nerf](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/A-Nerf)| 15.57 |0.508 |  0.242 | [ckpts](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/En56nksujH1Fn1qWiUJ-gpIBfzdHqHf66F-RvfzwTe2TBQ?e=Zz0EgX)|

(see detail why A-Nerf's performance is counterproductive in [issue](https://github.com/LemonATsu/A-NeRF/issues/8))
### Generalizable Methods on Genebody
| Model  | PSNR | SSIM |LPIPS| ckpts|
| :--- | :---------------:|:---------------:| :---------------:| :---------------:  |
| PixelNeRF | 24.15   |0.903 | 0.122 | |
| [IBRNet](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/ibrnet)| 23.61    |0.836 |  0.177 | [ckpts](https://hkustconnect-my.sharepoint.com/personal/wchengad_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fwchengad%5Fconnect%5Fust%5Fhk%2FDocuments%2Fgenebody%2Dbenchmark%2Dpretrained%2Fibrnet)|

## Citation

```
@article{cheng2022generalizable,
    title={Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    author={Cheng, Wei and Xu, Su and Piao, Jingtan and Qian, Chen and Wu, Wayne and Lin, Kwan-Yee and Li, Hongsheng},
    journal={arXiv preprint arXiv:2204.11798},
    year={2022}
}
```
