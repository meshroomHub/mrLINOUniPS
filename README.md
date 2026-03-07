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

### Inputs

| Parameter | Label | Description |
|-----------|-------|-------------|
| `inputSfm` | Input SfMData | SfMData JSON file with multi-lighting views grouped by poseId **(required)** |
| `maskFolder` | Mask Folder | Folder with mask PNGs named by poseId or viewId |
| `downscale` | Downscale Factor | Integer downscale factor for input images (1-8, default: 1) |
| `nbImages` | Number of Images | Number of lighting images per pose (-1 = all) |
| `useGpu` | Use GPU | Use GPU for inference (default: true) |

### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Folder containing normal map PNGs |
| `outputSfmDataNormal` | SfMData file referencing normal maps |

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
│       └── LINOUniPS.py           # Meshroom node definition
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
