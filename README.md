# Spherical View Synthesis for Self-Supervised 360<sup><b>o</b></sup> Depth Estimation

[![Paper](http://img.shields.io/badge/paper-arxiv.1909.08112-critical.svg?style=plastic)](https://arxiv.org/pdf/1909.08112.pdf)
[![Conference](http://img.shields.io/badge/3DV-2019-blue.svg?style=plastic)](http://3dv19.gel.ulaval.ca/)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/SphericalViewSynthesis/)
___

# Data
The 360<sup>o</sup> stereo data used to train the self-supervised models are available [here](https://vcl3d.github.io/3D60/) and are part of a larger dataset __\[[1](#OmniDepth), [2](#HyperSphere)\]__ that contains rendered color images, depth and normal maps for each viewpoint in a trinocular setup.

___

## Train
Training code to reproduce our experiments is available in this repository:

A set of training scripts are available for each different variant:

* [`train_ud.py`](./train_ud.py) for vertical stereo (__UD__) training
* [`train_lr.py`](./train_lr.py) for horizontal stereo (__LR__) training
* [`train_tc.py`](./train_tc.py) for trinocular stereo (__TC__) training, using the `photo_ratio` argument to train the different __TC__ variants.
* [`train_sv.py`](./train_sv.py) for supervised (__SV__) training

The PyTorch implementation of the differentiable depth-image-based forward rendering ([_`splatting`_](./supervision/splatting.py#L9)), presented in __\[[3](#LSI)\]__ and originally implemented in [TensorFlow](https://github.com/google/layered-scene-inference), is also [available](./supervision/splatting.py#L73).

## Test

Our evaluation script [`test.py`](./test.py) also includes the adaptation of the metrics calculation to spherical data that includes [spherical weighting](./spherical/weights.py#L8) and [spiral sampling](./test.py#L92).

## Pre-trained Models
Our PyTorch pre-trained models (corresponding to those reported in the paper) are available at our [releases](https://github.com/VCL3D/SphericalViewSynthesis/releases) and contain these model variants:

* [__UD__ @ epoch 16](https://github.com/VCL3D/SphericalViewSynthesis/releases/download/UD/ud.pt)
* [__TC8__ @ epoch 16](https://github.com/VCL3D/SphericalViewSynthesis/releases/download/TC8/tc8.pt)
* [__TC6__ @ epoch 28](https://github.com/VCL3D/SphericalViewSynthesis/releases/download/TC6/tc6.pt)
* [__TC4__ @ epoch 17](https://github.com/VCL3D/SphericalViewSynthesis/releases/download/TC4/tc4.pt)
* [__TC2__ @ epoch 20](https://github.com/VCL3D/SphericalViewSynthesis/releases/download/TC2/tc2.pt)
* [__LR__ @ epoch 18](https://github.com/VCL3D/SphericalViewSynthesis/releases/download/LR/lr.pt)
* [__SV__ @ epoch 24](https://github.com/VCL3D/SphericalViewSynthesis/releases/download/SV/sv.pt)

___

## Citation
If you use this code and/or data, please cite the following:
```
@inproceedings{zioulis2019spherical,
  author       = "Zioulis, Nikolaos and Karakottas, Antonis and Zarpalas, Dimitris and Alvarez, Federic and Daras, Petros",
  title        = "Spherical View Synthesis for Self-Supervised $360^o$ Depth Estimation",
  booktitle    = "International Conference on 3D Vision (3DV)",
  month        = "September",
  year         = "2019"
}
```


# References
<a name="OmniDepth"/>__\[[1](https://vcl.iti.gr/360-dataset)\]__ Zioulis, N.__\*__, Karakottas, A.__\*__, Zarpalas, D., and Daras, P. (2018). [Omnidepth: Dense depth estimation for indoors spherical panoramas](https://arxiv.org/pdf/1807.09620.pdf). In Proceedings of the European Conference on Computer Vision (ECCV).

<a name="HyperSphere"/>__\[[2](https://vcl3d.github.io/HyperSphereSurfaceRegression/)\]__ Karakottas, A., Zioulis, N., Samaras, S., Ataloglou, D., Gkitsas, V., Zarpalas, D., and Daras, P. (2019). [360<sup>o</sup> Surface Regression with a Hyper-sphere Loss](https://arxiv.org/pdf/1909.07043.pdf). In Proceedings of the International Conference on 3D Vision (3DV).

<a name="LSI"/>__[3]__ Tulsiani, S., Tucker, R., and Snavely, N. (2018). [Layer-structured 3d scene inference via view synthesis](https://arxiv.org/pdf/1807.10264.pdf). In Proceedings of the European Conference on Computer Vision (ECCV).
