<div align="center">

<h1 style="margin: 0;">UniTriSplat: A Unified 3D Gaussian Splatting Framework with Uniform Spherical Rasterization for Universal Cameras</h1>

<p style="margin: 8px 0 2px 0;">
  <a href="#paper"><img alt="Paper" src="https://img.shields.io/badge/Paper-Coming%20soon-blue?style=flat" /></a>
  <a href="#arxiv"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-Coming%20soon-green?style=flat" /></a>
  <a href="https://yipengzhu0809.github.io/UniTriSplat/"><img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-Website-orange?style=flat" /></a>
  <a href="#sub-hssim"><img alt="HSSIM" src="https://img.shields.io/badge/HSSIM-Coming%20soon-red?style=flat" /></a>
</p>

<p style="margin: 6px 0 0 0;">The Hong Kong University of Science and Technology</p>
<p style="margin: 6px 0 0 0;">Beijing Institute of Technology</p>
<p style="margin: 2px 0;">The 19th European Conference on Computer Vision -- ECCV 2026</p>

<p style="margin: 6px 0 0 0;">
  <a href="https://scholar.google.com/citations?user=pb3sxigAAAAJ&hl=zh-CN">Yipeng Zhu</a>,
  <a href="https://huajianup.github.io/">Huajian Huang<sup>†</sup></a>,
  <a href="https://seng.hkust.edu.hk/about/people/faculty/tristan-camille-braud">Tristan Braud</a>,
  <a href="https://saikit.org/index.html">Sai-Kit Yeung</a>
</p>

<sup>†</sup>Corresponding author

</div>

<div align="center">
    <img src="docs/assets/Pipeline.png" alt="UniTriSplat pipeline" width="95%">
    <br>
    <em>Pipeline of the UniTriSplat.</em>
    <br><br>
</div>


> **Abstract:**
Existing 3D Gaussian Splatting (3DGS) frameworks rely on camera-specific rasterization, suffering from inconsistent solid-angle sampling and degraded performance across heterogeneous camera models (e.g., perspective, fisheye, omnidirectional).
To address this limitation, we propose UniTriSplat, a unified 3DGS framework for universal cameras that reformulates Gaussian splatting on the unit sphere via HEALPix discretization.
Leveraging the equal-area property of HEALPix, we construct a spherical sampling grid aligned with the angular resolution of input images. We derive the forward rendering and gradient propagation of Gaussians directly in the spherical radian domain, yielding uniform optimization behavior from narrow-FoV images to full 360-degree panoramas.
To enhance perceptual reconstruction quality, we additionally introduce a HEALPix-aware SSIM loss that respects spherical neighborhood structure.
Extensive experiments across diverse camera models demonstrate that UniTriSplat consistently improves cross-camera generalization while preserving geometric fidelity and rendering quality.

<!-- 可选：用于 badge 占位的锚点 -->
<span id="paper"></span>
<span id="arxiv"></span>
<span id="project-page"></span>
<span id="sub-hssim"></span>

## Updates:
- 🚧 **Coming soon:** Full source code release.
- 🚧 **Coming soon:** HSSIM submodule source code release.
- 🚧 **Coming soon:** Project page.
- ✅ **2026-06-17:** UniTriSplat was accepted to ECCV 2026! 🎉
