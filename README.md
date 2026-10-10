<div align="center">

# mrLINOUniPS

### Meshroom Plugin for LINO_UniPS

<p>
Integrate <a href="https://github.com/meshroomHubWarehouse/LINO_UniPS">LINO_UniPS</a> photometric stereo normal (and albedo) estimation directly into your <a href="https://github.com/alicevision/Meshroom">Meshroom</a> photogrammetry pipeline.
</p>

<a href="https://github.com/meshroomHubWarehouse/LINO_UniPS"><img src="https://img.shields.io/badge/Core-LINO__UniPS-green" alt="LINO_UniPS" height="25"></a>

</div>

---

## What is LINO_UniPS?

**LINO_UniPS** is a universal photometric stereo method based on a light-invariant normal estimator. It predicts high-quality per-pixel surface normals from multi-lighting images without requiring known light directions. The method leverages diffusion model priors and handles arbitrary numbers of input images with varying illumination conditions.

A second network, **LINO-UniPS PBR** (`lino_pbr.pth`), adds base color, roughness and metallic heads to the same
features: the node uses its base color as the albedo map of each pose (`Estimate Albedo`), e.g. for the albedo input
of RNb-NeuS2.

---

## Requirements

- **Python** 3.10+
- **CUDA** 12.x + NVIDIA GPU (ampere or newer recommended for bfloat16 support)
- **[Meshroom](https://github.com/alicevision/Meshroom)** 2025+ (develop branch)

---

## Quick Start

> **Prerequisite:** a working [Meshroom](https://github.com/alicevision/Meshroom) installation.

### 1. Clone the plugin

```bash
cd /path/to/your/plugins
git clone https://github.com/meshroomHub/mrLINOUniPS.git
cd mrLINOUniPS
```

### 2. Set up the virtual environment

Meshroom looks for a folder named **`venv`** at the plugin root.

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision
pip install -e .

deactivate
```

The plugin's `pyproject.toml` installs LINO_UniPS and its dependencies. Meshroom
still discovers the node from this repository via `MESHROOM_PLUGINS_PATH`;
`requirements.txt` is retained for older installation workflows.

### 3. Download pretrained weights

```bash
bash download_weights.sh        # lino.pth: normal model
bash download_weights.sh pbr    # lino_pbr.pth: PBR model, needed for 'Estimate Albedo' (or 'Normal Model' = pbr)
bash download_weights.sh all    # both
```

This downloads the pretrained models (~338 MB each) from HuggingFace
([houyuanchen/lino](https://huggingface.co/houyuanchen/lino)) into `weights/`:

```
weights/
├── lino.pth
└── lino_pbr.pth
```

The plugin auto-detects these files. No config.json needed.

### 4. Register the plugin in Meshroom

```bash
export MESHROOM_PLUGINS_PATH=/path/to/your/plugins/mrLINOUniPS:$MESHROOM_PLUGINS_PATH
```

Launch Meshroom: the **LINOUniPS** node appears under **Photometric Stereo**.

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
| `modelPath` | | Weights of the normal model; default: `weights/lino.pth` of the plugin, then of LINO_UniPS, then the torch hub cache |
| `pbrModelPath` | | Weights of the PBR model (albedo or `pbr` normals); default: `weights/lino_pbr.pth`, same search order |
| `linoUniPsPath` | `${LINO_UNIPS_PATH}` | LINO_UniPS code, used when the package is not installed in the plugin environment |

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

## Advanced: Developer Setup

If you prefer to work from a local LINO_UniPS clone instead of pip install:

1. Clone the repo: `git clone https://github.com/meshroomHubWarehouse/LINO_UniPS.git`
2. Edit `meshroom/config.json`:
   ```json
   [
       {"key": "LINO_UNIPS_PATH", "type": "path", "value": "/path/to/LINO_UniPS"}
   ]
   ```
3. Place `lino.pth` (and `lino_pbr.pth`) in the `weights/` subdirectory of LINO_UniPS.

The node searches for `lino.pth` (and `lino_pbr.pth`) in this order, unless `modelPath` (`pbrModelPath`) is set:
1. Plugin `weights/` directory
2. `weights/` of the LINO_UniPS code directory (from config.json)
3. Torch hub cache (`~/.cache/torch/hub/checkpoints/`)

---

## Plugin Structure

```
mrLINOUniPS/
├── meshroom/
│   ├── config.json                # Plugin configuration (optional for dev)
│   └── LINOUniPS/
│       ├── __init__.py
│       ├── LINOUniPS.py           # Meshroom node definition
│       └── psCommon.py            # Common photometric stereo layer (identical in mrUniMSPS, mrSDMUniPS)
├── tests/                         # pytest tests (CPU) + check_real_pose.py (GPU check on a real pose)
├── weights/                       # Downloaded model weights
│   ├── lino.pth
│   └── lino_pbr.pth
├── venv/                          # Python virtual environment
├── download_weights.sh            # Weight download script
├── pyproject.toml                 # Plugin dependency metadata
├── requirements.txt               # Legacy dependency list
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
| [mrOpenRNb](https://github.com/meshroomHub/mrOpenRNb) | Meshroom plugin for neural surface reconstruction from normals |

---

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
