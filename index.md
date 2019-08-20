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

![Network & Supervision](./assets/images/network.png "CNN architecture & supervision schemes")

![Spherical Disparity Model](./assets/images/spherical_disparity_model.png "Geometrically Derived Spherical Disparity Model")

# Paper
<a href="https://arxiv.org">
  <img src="./assets/images/paper_thumb.png" width="200">
</a>

# Data

# Models

# Citation
<p style="
    width: auto;
    background-color: #f2f2f2;
    font-size: small;
">
  <pre>
    <code>
      @inproceedings{
        karakottas2019Surface
      }
    </code>
  </pre>
</p>
# References
