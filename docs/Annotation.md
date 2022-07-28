# GeneBody Annotations

## Data Capture
GeneBody dataset captures performer in a motion capture studio with 48 synchronized cameras. Each actor is asked to perform 10 seconds clips recorded in a 15 fps rate. The camera location and capture volume is visualized in the following video.
<!-- ![Teaser image](./genebody.gif#center) -->
<p align="center"><img src="./capture_volume.gif" width="90%"></p>
<p align="center">Left: Motion capture studio and performer, cameras are highlighted. Right: Video captured from camera 25.</p>


## Dataset Organization
The processed GeneBody dataset is organized in following structure
```
├──genebody/            # root of dataset
  ├──amanda/            # subject
    ├──image/           # multiview images
        ├──00/          # images of 00 view
        ├──...
    ├──mask/            # multiview masks
        ├──00/          # masks of 00 view
        ├──...
    ├──param/           # smpl parameters
    ├──smpl/            # smpl meshes in OBJ format
    ├──annots.npy       # camera parameters
  ├──.../
  ├──genebody_split.npy # dataset splits
```
You can download the Test10 and Train40 subset by the [instructions](./Dataset.md#download-instructions). 


## Data Interface
We provide the reference data reader `GeneBodyReader` in `genebody/genebody.py`.
### Source views
The default source view number is 4, and the source views are `[01, 13, 25, 37]`.
### Image cropping
As human performer may appear in different size across views, and the original image plane contains very small proportion of foreground, directly apply image quality metrics on raw image, eg. PSNR, SSIM and LPIPS may introduce ambiguity numerically. To tackle this  we crop the performer out and resize it to a give resolution, in GNR and GeneBody benchmarks. Check the reference interface for more details.

## Camera Calibration
GeneBody provides the camera calibration for each subjects, intrinic matrix, distortion coefficient and extrinic parameters are provided in `annots.npy` in each subject folder. Note that we use a opencv camera (xyz->right,down,front).

## Human Segmentation
### Auto annotation
Before recording, we capture the scene without any performer, and use it as reference image in [BackgroundMatting-V2](https://github.com/PeterL1n/BackgroundMattingV2) to automatically extract the performer foreground mask.
### Human labeling
We choose 8 camera views to manually check the results of auto annoataion and manually labels the bad case. The 8 camera views are `[01, 07, 13, 19, 25, 31, 37, 43]`.

## SMPLx
GeneBody provides per-frame SMPLx estimation, and store the mesh in `smpl` subfolder and SMPLx parameters in `param` subfolder. The SMPLx toolbox is also provided in this [repo](https://github.com/generalizable-neural-performer/bodyfitting).

### SMPLx parameters
GeneBody provide SMPLs parameter and 3D keypoints in `param` subfolder. More specifally, the dictionary 'smplx', can be directly feed to SMPLX [forward](https://github.com/vchoutas/smplx/blob/master/smplx/body_models.py#L1111) pass as long as all the parameters are converted to torch tensor, an example is provided in the [data interface](../genebody/genebody.py#L189).

### SMPLx scale
GeneBody has a large variation on performers' age distribution, while SMPLx model typically fails to fit well on kids and giants. We introduce 'smplx_scale' outside the SMPLx model, and jointly optimize scale and body model parameters during fitting. Thus, to recover the fitted mesh in `smpl` subfolder using parameters in `param` subfolder, you need to multiple a 'smplx_scale' to the vertices or 3d joints output by the body model.