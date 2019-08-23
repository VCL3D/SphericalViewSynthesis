![Omnidirectional Trinocular Stereo](./assets/images/trinocular_3d_scene.png "Omnidirectional Trinocular Stereo")

# Abstract

Learning based approaches for depth perception are limited by the availability of clean training data. 
This has led to the utilization of view synthesis as an indirect objective for learning depth estimation using efficient data acquisition procedures. 
Nonetheless, most research focuses on pinhole based monocular vision, with scarce works presenting results for omnidirectional input.
In this work, we explore spherical view synthesis for learning monocular 360<sup>o</sup> depth in a self-supervised manner and demonstrate its feasibility.
Under a purely geometrically derived formulation we present results for horizontal and vertical baselines, as well as for the trinocular case.
Further, we show how to better exploit the expressiveness of traditional CNNs when applied to the equirectangular domain in an efficient manner.
Finally, given the availability of ground truth depth data, our work is uniquely positioned to compare view synthesis against direct supervision in a consistent and fair manner.
The results indicate that alternative research directions might be better suited to enable higher quality depth perception.
Our data, models and code are publicly available at [our project page](https://vcl3d.github.io/SphericalViewSynthesis/).

# Model
![Spherical & Cartesian Coordinates](./assets/images/spherical_cartesian.png "Spherical & Cartesian Coordinates")
![Partial Derivatives of Spherical to Cartesian Coordinates](./assets/images/spherical_derivatives.png "Partial Derivatives of Spherical to Cartesian Coordinates")
![Spherical Disparity Model](./assets/images/spherical_disparity_model.png "Geometrically Derived Spherical Disparity Model")

## Code
![Network & Supervision](./assets/images/network.png "CNN architecture & supervision schemes")

## Pre-trained Models
Coming Soon...

# Data
Coming Soon...

# Publication
## Paper
<!--
[![arXiv paper link](./assets/images/paper_all_pages.png "arXiv")](https://arxiv.org)
-->
<a href="https://arxiv.org"><img src="./assets/images/paper_all_pages.png" width="700" title="arXiv paper link" alt="arXiv"/></a>

## Supplementary
<!--
[![local supplementary link](./assets/images/supplementary_all_pages.png "arXiv")](https://arxiv.org)
-->
<a href="https://arxiv.org"><img src="./assets/images/supplementary_all_pages.png" width="700" title="supplementary link" alt="arXiv"/></a>

## Authors
[Nikolaos Zioulis](zokin.github.io), [Antonis Karakottas](https://ankarako.github.io/), [Dimitris Zarpalas](https://www.iti.gr/iti/people/Dimitrios_Zarpalas.html), [Federico Alvarez](https://www.researchgate.net/profile/Federico_Alvarez3) and [Petros Daras](https://www.iti.gr/iti/people/Petros_Daras.html)

[Visual Computing Lab (VCL)](http://vcl.iti.gr/)

## Citation
If you use this code and/or data, please cite the following:
```
@inproceedings{zioulis2019spherical,
  author       = "Zioulis, Nikolaos and Karakottas, Antonis and Zarpalas, Dimitris and Alvarez, Federic and Daras, Petros",
  title        = "Spherical View Synthesis for Self-Supervised $360^o$ Depth Estimation",
  booktitle    = "International Conference on 3D Vision (3DV)",
  year         = "2019"
}
```

## Acknowledgements
We thank the anonymous reviewers for helpful comments.

This project has received funding from the European Union’s Horizon 2020 research and innovation programme [Hyper360](http://hyper360.eu/) under grant agreement No 761934.

<!--
We also gratefully acknowledge NVIDIA corporation for the donation of a NVIDIA Titan X GPU used for this research. 
-->
We would like to thank NVIDIA for supporting our research with the donation of a Titan Xp GPU through the NVIDIA GPU Grant Program.

<img src="./assets/images/en_square_cef_logo_0.png" width="150"/><img src="./assets/images/h360.png" width="150"/><img src="./assets/images/NVLogo_2D.jpg" width="150"/>

<!--
![EC Funding  Logo](./assets/images/en_square_cef_logo_0.png "EC Funding Logo")
![Hyper360 Logo](./assets/images/h360.png "Hyper360 Logo")
![NVIDIA Logo](./assets/images/NVLogo_2D.jpg "NVIDIA Logo")
-->

# Contact
Please direct any questions related to the code, models and dataset to nzioulis@iti.gr or post a [GitHub issue](https://github.com/VCL3D/SphericalViewSynthesis/issues).

# References
Coming Soon...
