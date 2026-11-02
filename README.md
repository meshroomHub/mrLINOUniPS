<div align="center">

# mrLINOUniPS

### Meshroom Plugin for LINO-UniPS

<p>
Integrate <a href="https://github.com/meshroomHubWarehouse/LINO_UniPS">LINO-UniPS</a> photometric stereo normal (and albedo) estimation directly into your <a href="https://github.com/alicevision/Meshroom">Meshroom</a> photogrammetry pipeline.
</p>

<a href="https://github.com/meshroomHubWarehouse/LINO_UniPS"><img src="https://img.shields.io/badge/Core-LINO--UniPS-green" alt="LINO-UniPS" height="25"></a>

</div>

---

## What is LINO-UniPS?

**LINO-UniPS** is a universal photometric stereo method based on a light-invariant normal estimator. It predicts high-quality per-pixel surface normals from multi-lighting images without requiring known light directions. The method leverages diffusion model priors and handles arbitrary numbers of input images with varying illumination conditions.

A second network, **LINO-UniPS PBR** (`lino_pbr.pth`), adds base color, roughness and metallic heads to the same
features: the node uses its base color as the albedo map of each pose (`Estimate Albedo`), e.g. for the albedo input
of RNb-NeuS2.

---

## Requirements

- **[Meshroom](https://github.com/alicevision/Meshroom)** 2026.1 or later (develop branch until its release):
  Meshroom 2025.1.x has neither the plugin manager nor the `[tool.meshroom]` settings of `pyproject.toml`
- **NVIDIA GPU** Ampere or newer (compute capability 8.0 or higher, for bfloat16) with a driver supporting CUDA 12.1; older GPUs
  fail ([#4](https://github.com/meshroomHub/mrLINOUniPS/issues/4)): disable `Use GPU` to run on the CPU (slower)
- **Python** 3.10 or newer

All Python dependencies are declared in [`pyproject.toml`](pyproject.toml): PyTorch (CUDA 12.1) and `lino-unips`, pinned to the
`weights-v1` tag of [meshroomHubWarehouse/LINO_UniPS](https://github.com/meshroomHubWarehouse/LINO_UniPS), a fork of
[houyuanchen111/LINO_UniPS](https://github.com/houyuanchen111/LINO_UniPS).

---

## Installation

### With the plugin manager

From Meshroom's plugin manager (*Manage Plugins*), or from the command line:

```bash
meshroom_plugins install mrLINOUniPS
```

The installer creates the plugin `venv/` with `uv sync` and downloads the normal and PBR models (~354 MB each), declared as assets in
`pyproject.toml`, checking their SHA-256. The **LINOUniPS** node then appears under the **Photometric Stereo** category.

### Manual install (without the plugin manager)

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd /path/to/plugins
git clone https://github.com/meshroomHub/mrLINOUniPS.git
cd mrLINOUniPS
UV_PROJECT_ENVIRONMENT=venv uv sync   # Meshroom looks for venv/ at the plugin root

URL=https://github.com/meshroomHubWarehouse/LINO_UniPS/releases/download/weights-v1
mkdir -p weights
curl -L -o weights/lino.pth $URL/lino.pth
curl -L -o weights/lino_pbr.pth $URL/lino_pbr.pth
echo "7b422cb73474e74758907b5cfd8585b5fe2ffc87a84d79bb3dadb20a49134fb1  weights/lino.pth" | sha256sum -c
echo "b02dd1726093011d47ad2b2390245af7ea0e6c282cc3bf9a20e06a5964c1757a  weights/lino_pbr.pth" | sha256sum -c

export MESHROOM_PLUGINS_PATH=/path/to/plugins/mrLINOUniPS:$MESHROOM_PLUGINS_PATH
```

On Windows (PowerShell): `$env:UV_PROJECT_ENVIRONMENT = "venv"; uv sync`, then
`$env:MESHROOM_PLUGINS_PATH = "C:\path\to\plugins\mrLINOUniPS"`.

`lino_pbr.pth` is needed for 'Estimate Albedo' (or 'Normal Model' = pbr). The weights mirror the models of
[houyuanchen/lino](https://huggingface.co/houyuanchen/lino) on Hugging Face; the plugin finds them in `weights/`.

### Working on a local LINO-UniPS clone

The node imports LINO-UniPS from the plugin `venv/`. To run a local clone instead, install it in editable mode in that venv:

```bash
uv pip install --python venv/bin/python -e /path/to/LINO_UniPS
```

Updating or reinstalling the plugin from the plugin manager recreates `venv/`: run this command again afterwards.
`LINO_UNIPS_PATH` (an environment variable, or the `linoUniPsPath` input of the node) is only a fallback: the node uses it when
LINO-UniPS is not installed in the venv.

The node searches for `lino.pth` (and `lino_pbr.pth`) in this order, unless `modelPath` (`pbrModelPath`) is set:
1. Plugin `weights/` directory
2. `weights/` of the LINO-UniPS code directory (`LINO_UNIPS_PATH`)
3. Torch hub cache (`~/.cache/torch/hub/checkpoints/`)

---

## Pipeline template

`meshroom/linoUniPs.mgt` (*File > New Pipeline > Lino Uni Ps*) runs the structure-from-motion of AliceVision, segments the
object with SAM 3, exports the undistorted images and estimates the normals with `LINOUniPS`.

It needs [mrSegmentation](https://github.com/meshroomHub/mrSegmentation) (node `ImageSegmentationSam3`), not in the
plugin registry yet: install it by hand.

Images: one folder per pose, holding the single-light images and one reference image lit by all the lights, whose
filename contains `ambient` (CameraInit makes it the pose reference) and `all_leds` (`SfMFilter_1` keeps these images
for the structure from motion and the segmentation). Set the SAM 3 prompt (`ImageSegmentationSam3_1`, default `stone`)
to your object.

---

## Node Parameters

The data handling (SfMData, image selection, masks, outputs) is implemented in `psCommon.py`, a common layer shared
as an identical copy by the LINOUniPS, UniMSPS and SDMUniPS nodes: the three nodes share these inputs, outputs and
behaviour, only the network and its specific options (below) differ.

### Inputs

| Parameter | Label | Default | Description |
|-----------|-------|---------|-------------|
| `inputSfm` | SfMData | | SfMData whose views sharing a poseId are the lighting images of a pose (e.g. ExportImages output) **(required)** |
| `maskFolder` | Mask Folder | | Masks `<poseId>.png` (one per pose) or `<viewId>.png` (one per image, combined by vote); alpha channels otherwise |
| `downscale` | Downscale Factor | 1 | Integer downscale factor of the images (and of the output maps and intrinsics) |
| `nbImages` | Number Of Images | -1 | Maximum number of lighting images per pose (-1: all); the GPU memory grows with it |
| `outputFormat` | Output Format | `png16` | `png16` (16-bit PNG, (n + 1) / 2) or `exr` (float32) |
| `useGpu` | Use GPU | true | Use the GPU (CPU otherwise) |

Inputs specific to LINO-UniPS (see [Albedo](#albedo)):

| Parameter | Label | Default | Description |
|-----------|-------|---------|-------------|
| `estimateAlbedo` | Estimate Albedo | false | Also output one albedo map per pose: base color of the LINO-UniPS PBR network (`lino_pbr.pth`) |
| `normalModel` | Normal Model | `normal` | Network of the normals: `normal` (LINO-UniPS, `lino.pth`) or `pbr` (LINO-UniPS PBR, `lino_pbr.pth`) |

Advanced inputs, common to the three nodes:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `minViewsPerPose` | 3 | Minimum number of views of a pose to process it (photogrammetry views are ignored) |
| `imageSelection` | `random` | Choice of the images when `nbImages` is lower than the number of images: `random`, `uniform`, `first` |
| `seed` | 42 | Seed of the random selection and of the network, combined with the poseId (reproducible per pose) |
| `linearizeInput` | false | Convert 8/16-bit images from sRGB to linear values |
| `maskThreshold` | 0.5 | Binarization of the masks and alpha channels, as a fraction of the value range |
| `maskVoteThreshold` | 0.5 | A pixel is in the pose mask when the fraction of per-image masks containing it is greater than this value (0.5: strict majority, 1: intersection, 0: union) |
| `maskRemoveBorderComponents` | true | Remove the alpha mask components touching the image border (valid area of undistorted images) |
| `maskUseGlobalFile` | false | Use `<maskFolder>/mask.png` for the poses without a specific mask |
| `normalConvention` | `opengl` | Output frame: `opengl` (x right, y up, z towards the camera, expected by RNb-NeuS2) or `opencv` |
| `keepLandmarks` | true | Keep the 3D landmarks in the output SfMData |
| `failurePolicy` | `noPose` | Fail if no pose could be processed (`noPose`), if any pose failed (`anyPose`), or `never` |

Advanced inputs specific to LINO-UniPS:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cropMargin` | 8 | Margin around the mask bounding box (the whole image is used when the box is closer to the border) |
| `maxProcessingSize` | 6000 | Maximum side of the square network input (multiple of 512) |
| `outputInterpolation` | `cubic` | Resampling of the prediction back to the crop size: `area`, `linear`, `cubic` |
| `modelPath` | | Weights of the normal model; default: `weights/lino.pth` of the plugin, then of LINO-UniPS, then the torch hub cache |
| `pbrModelPath` | | Weights of the PBR model (albedo or `pbr` normals); default: `weights/lino_pbr.pth`, same search order |
| `linoUniPsPath` | `${LINO_UNIPS_PATH}` | LINO-UniPS code, used when the package is not installed in the plugin environment |

### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Normal maps `<poseId>.png` (or `.exr`), pose masks `masks/<poseId>.png`, albedo maps `albedo/<poseId>.png` (or `.exr`) |
| `outputSfmDataNormal` | SfMData referencing the normal maps: one view per pose (the view whose viewId is the poseId), intrinsics scaled by `downscale` |
| `outputSfmDataAlbedo` | With `estimateAlbedo`: SfMData referencing the albedo maps (same views and intrinsics as the normal maps), for the `Albedo Maps SfMData` input of RNb-NeuS2 |
| `outputMaskFolder` | Pose masks (0/255): the pixels where a normal is defined |

Downscaling keeps the camera model exact: the downscaled pixel `i` averages the input pixels `[i * d, (i + 1) * d)`,
so the principal point becomes `(pp + 0.5) / d - 0.5` (AliceVision puts the center of pixel `i` at `i`).

### Albedo

With `estimateAlbedo`, the albedo of each pose is the base color predicted by the LINO-UniPS PBR network, in [0, 1]
(as predicted: no color space conversion), zero outside the pose mask, with the size and format (`png16`: 16-bit
RGB, `exr`: float32) of the normal maps. Like any photometric stereo albedo under unknown lighting, its scale is
not physical (RNb-NeuS2 rescales the albedos of the views).

`normalModel` chooses the network of the normals:

| `normalModel` | `estimateAlbedo` | Networks run per pose | Normals | Albedo |
|---------------|------------------|-----------------------|---------|--------|
| `normal` (default) | false (default) | LINO-UniPS | LINO-UniPS | |
| `normal` | true | LINO-UniPS + LINO-UniPS PBR (two passes) | LINO-UniPS | LINO-UniPS PBR |
| `pbr` | true | LINO-UniPS PBR (one pass) | LINO-UniPS PBR | LINO-UniPS PBR |
| `pbr` | false | LINO-UniPS PBR | LINO-UniPS PBR | |

The PBR network has the architecture of the normal network plus the base color, roughness and metallic heads, with
its own weights (all of them differ from `lino.pth`). Its normals are not evaluated on the photometric stereo
benchmarks by the authors, hence `normal` stays the default.

At their default values, `estimateAlbedo` and `normalModel` do not enter the node UID (and `pbrModelPath` is
disabled): the LINOUniPS nodes of existing projects keep their UID and computed caches.

---

## Tests

```bash
# unit tests (CPU), from the plugin root
PYTHONPATH=/path/to/Meshroom venv/bin/python -m pytest tests
# one real pose (GPU): checks sizes, mask coverage and the OpenGL frame of the normals
PYTHONPATH=/path/to/Meshroom venv/bin/python tests/check_real_pose.py <sfm> <poseId> <outputFolder> --downscale 2
```

---

## Plugin Structure

```
mrLINOUniPS/
├── meshroom/
│   ├── linoUniPs.mgt              # Pipeline template
│   └── LINOUniPS/
│       ├── __init__.py
│       ├── LINOUniPS.py           # Meshroom node definition
│       └── psCommon.py            # Common photometric stereo layer (identical in mrUniMSPS, mrSDMUniPS)
├── tests/                         # pytest tests (CPU) + check_real_pose.py (GPU check on a real pose)
├── weights/                       # Model weights (Meshroom asset)
│   ├── lino.pth
│   └── lino_pbr.pth
├── venv/                          # Created by the installer or by uv sync
├── pyproject.toml                 # Dependencies and Meshroom settings (uv)
└── README.md
```

For more details on how Meshroom plugins work, see:
- [Meshroom Plugin Install Guide](https://github.com/alicevision/Meshroom/blob/develop/INSTALL_PLUGINS.md)
- [mrHelloWorld](https://github.com/meshroomHub/mrHelloWorld): step-by-step tutorials for building Meshroom plugins

---

## Acknowledgements

This work is supported by [**DOPAMIn**](https://www.cnrsinnovation.com/actualite/une-seconde-promotion-pour-le-programme-open-7-nouveaux-logiciels-scientifiques-a-valoriser/) (*Diffusion Open de Photogrammetrie par AliceVision/Meshroom pour l'Industrie*), selected in the 2024 cohort of the [**OPEN**](https://www.cnrsinnovation.com/open/) programme run by [CNRS Innovation](https://www.cnrsinnovation.com/). OPEN supports the valorization of open-source scientific software by providing dedicated developer resources, governance expertise, and industry partnership support.

**Lead researcher:** [Jean-Denis Durou](https://cv.hal.science/jean-denis-durou), [IRIT](https://www.irit.fr/) (INP-Toulouse)
**Co-lead:** [Lilian Calvet](https://fr.linkedin.com/in/lilian-calvet-42b1a689), [Balgrist University Hospital](https://www.balgrist.ch/)

---

## Related Projects

| Project | Description |
|---------|-------------|
| [LINO_UniPS](https://github.com/meshroomHubWarehouse/LINO_UniPS) | Light-invariant normal estimator for universal photometric stereo |
| [mrSDMUniPS](https://github.com/meshroomHub/mrSDMUniPS) | Meshroom plugin for SDM-UniPS photometric stereo |
| [mrUniMSPS](https://github.com/meshroomHub/mrUniMSPS) | Meshroom plugin for Uni-MS-PS photometric stereo |
| [mrLedGetter](https://github.com/meshroomHub/mrLedGetter) | Meshroom plugin for LED-getter near-light photometric stereo |
| [mrRNbNeuS2](https://github.com/meshroomHub/mrRNbNeuS2) | Meshroom plugin for RNb-NeuS2 neural surface reconstruction |
| [mrOpenRNb](https://github.com/meshroomHub/mrOpenRNb) | Meshroom plugin for Open-RNb neural surface reconstruction |

---

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
