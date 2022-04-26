# Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis
[![report](https://img.shields.io/badge/arxiv-report-red)]() 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

![Teaser image](./docs/teaser.png)

> **Abstract:** *This work targets at using a general deep learning framework to synthesize free-viewpoint images of arbitrary human performers, only requiring a sparse number of camera views as inputs and skirting per-case fine-tuning. The large variation of geometry and appearance, caused by articulated body poses, shapes and clothing types, are the key bot tlenecks of this task. To overcome these challenges, we present a simple yet powerful framework, named Generalizable Neural Performer (GNR), that learns a generalizable and robust neural body representation over various geometry and appearance. Specifically, we compress the light fields for novel view human rendering as conditional implicit neural radiance fields with several designs from both geometry and appearance aspects. We first introduce an Implicit Geometric Body Embedding strategy to enhance the robustness based on both parametric 3D human body model prior and multi-view source images hints. On the top of this, we further propose a Screen-Space Occlusion-Aware Appearance Blending technique to preserve the high-quality appearance, through interpolating source view appearance to the radiance fields with a relax but approximate geometric guidance.* <br>

Wei Cheng, Su Xu, Jingtan Piao, Chen Qian, Wayne Wu, Kwan-Yee Lin, Hongsheng Li<br>
**[[Demo Video]](https://www.youtube.com/watch?v=2COR4u1ZIuk)** | **[[Project Page]](https://generalizable-neural-performer.github.io/)** | **[[Data]](https://generalizable-neural-performer.github.io/genebody.html)** | **[[Paper]](https://arxiv.org/pdf/2204.11798.pdf)**

## Updates

- [26/04/2022] Code is coming soon!
- [26/04/2022] Part of data released!
- [26/04/2022] Techincal report released.
- [24/04/2022] The codebase and project page are created.


## Citation

```
@article{
    author = {Wei, Cheng and Su, Xu and Jingtan, Piao and Wayne, Wu and Chen, Qian and Kwan-Yee, Lin and Hongsheng, Li},
    title = {Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    publisher = {arXiv},
    year = {2022},
  }
```
