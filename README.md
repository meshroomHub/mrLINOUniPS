<div align="center">

# mrLINOUniPS

### Meshroom Plugin for LINO_UniPS

<p>
Integrate <a href="https://github.com/meshroomHubWarehouse/LINO_UniPS">LINO_UniPS</a> photometric stereo normal estimation directly into your <a href="https://github.com/alicevision/Meshroom">Meshroom</a> photogrammetry pipeline.
</p>

<a href="https://github.com/meshroomHubWarehouse/LINO_UniPS"><img src="https://img.shields.io/badge/Core-LINO__UniPS-green" alt="LINO_UniPS" height="25"></a>

</div>

---

## What is LINO_UniPS?

**LINO_UniPS** is a universal photometric stereo method based on a light-invariant normal estimator. It predicts high-quality per-pixel surface normals from multi-lighting images without requiring known light directions. The method leverages diffusion model priors and handles arbitrary numbers of input images with varying illumination conditions.

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
pip install -r requirements.txt

deactivate
```

This installs LINO_UniPS and all its dependencies automatically via pip.

### 3. Download pretrained weights

```bash
bash download_weights.sh
```

This downloads the pretrained model (~338 MB) from HuggingFace into `weights/`:

```
weights/
└── lino.pth
```

The plugin auto-detects this file. No config.json needed.

### 4. Register the plugin in Meshroom

```bash
export MESHROOM_PLUGINS_PATH=/path/to/your/plugins/mrLINOUniPS:$MESHROOM_PLUGINS_PATH
```

Launch Meshroom: the **LINOUniPS** node appears under **Photometric Stereo**.

---

## Node Parameters

The data handling (SfMData, image selection, masks, outputs) is implemented in `psCommon.py`, a common layer shared
as an identical copy by the LINOUniPS, UniMSPS and SDMUniPS nodes: the three nodes have the same inputs, outputs and
behaviour, only the network differs.

### Inputs

| Parameter | Label | Default | Description |
|-----------|-------|---------|-------------|
| `inputSfm` | SfMData | | SfMData whose views sharing a poseId are the lighting images of a pose (e.g. ExportImages output) **(required)** |
| `maskFolder` | Mask Folder | | Masks `<poseId>.png` (one per pose) or `<viewId>.png` (one per image, combined by vote); alpha channels otherwise |
| `downscale` | Downscale Factor | 1 | Integer downscale factor of the images (and of the output maps and intrinsics) |
| `nbImages` | Number Of Images | -1 | Maximum number of lighting images per pose (-1: all); the GPU memory grows with it |
| `outputFormat` | Output Format | `png16` | `png16` (16-bit PNG, (n + 1) / 2) or `exr` (float32) |
| `useGpu` | Use GPU | true | Use the GPU (CPU otherwise) |

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
| `modelPath` | | Weights; default: `weights/lino.pth` of the plugin, then of LINO_UniPS, then the torch hub cache |
| `linoUniPsPath` | `${LINO_UNIPS_PATH}` | LINO_UniPS code, used when the package is not installed in the plugin environment |

### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Normal maps `<poseId>.png` (or `.exr`), pose masks `masks/<poseId>.png` |
| `outputSfmDataNormal` | SfMData referencing the normal maps: one view per pose (the view whose viewId is the poseId), intrinsics scaled by `downscale` |
| `outputMaskFolder` | Pose masks (0/255): the pixels where a normal is defined |

Downscaling keeps the camera model exact: the downscaled pixel `i` averages the input pixels `[i * d, (i + 1) * d)`,
so the principal point becomes `(pp + 0.5) / d - 0.5` (AliceVision puts the center of pixel `i` at `i`).

---

## Advanced: Developer Setup

If you prefer to work from a local LINO_UniPS clone instead of pip install:

1. Clone the repo: `git clone -b meshroom https://github.com/meshroomHubWarehouse/LINO_UniPS.git`
2. Edit `meshroom/config.json`:
   ```json
   [
       {"key": "LINO_UNIPS_PATH", "type": "path", "value": "/path/to/LINO_UniPS"}
   ]
   ```
3. Place `lino.pth` in the LINO_UniPS directory or its `weights/` subdirectory.

The node searches for weights in this order:
1. Plugin `weights/` directory
2. LINO_UniPS code directory (from config.json)
3. Torch hub cache (`~/.cache/torch/hub/checkpoints/lino.pth`)

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
│   └── lino.pth
├── venv/                          # Python virtual environment
├── download_weights.sh            # Weight download script
├── requirements.txt               # Python dependencies (pip install from git)
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
