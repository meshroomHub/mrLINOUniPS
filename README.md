<div align="center">

# mrLINOUniPS

### Meshroom Plugin for LINO_UniPS

<p>
Integrate <a href="https://github.com/houyuanchen111/LINO_UniPS">LINO_UniPS</a> photometric stereo normal estimation directly into your <a href="https://github.com/alicevision/Meshroom">Meshroom</a> photogrammetry pipeline.
</p>

<a href="https://github.com/houyuanchen111/LINO_UniPS"><img src="https://img.shields.io/badge/Core-LINO__UniPS-green" alt="LINO_UniPS" height="25"></a>

</div>

---

## What is LINO_UniPS?

**LINO_UniPS** is a universal photometric stereo method based on a light-invariant normal estimator. It predicts high-quality per-pixel surface normals from multi-lighting images without requiring known light directions. The method leverages diffusion model priors and handles arbitrary numbers of input images with varying illumination conditions.

---

## Requirements

- **Python** 3.10+
- **CUDA** 12.x + NVIDIA GPU (ampere or newer recommended for bfloat16 support)
- **[Meshroom](https://github.com/alicevision/Meshroom)** 2025+ (develop branch)

Full dependency list: [`requirements.txt`](requirements.txt)

---

## Quick Start

> **Prerequisite:** a working [Meshroom](https://github.com/alicevision/Meshroom) installation.

### 1. Clone the plugin

```bash
cd /path/to/your/plugins
git clone https://github.com/meshroomHub/mrLINOUniPS.git
```

### 2. Clone the LINO_UniPS core code

```bash
git clone https://github.com/houyuanchen111/LINO_UniPS.git
```

> **Note:** for SfMData JSON support, use the `feat/meshroom-plugin` branch.

### 3. Set up the virtual environment

Meshroom looks for a folder named **`venv`** at the plugin root and uses its Python interpreter to run the node. You have two options:

#### Option A: Symlink an existing venv

If you already have a working virtual environment from the LINO_UniPS repository, you can simply symlink it:

```bash
cd mrLINOUniPS
ln -s /absolute/path/to/LINO_UniPS/.venv venv
```

#### Option B: Create a fresh venv

```bash
cd mrLINOUniPS

# Create the venv (must be named "venv", not ".venv")
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt

deactivate
```

### 4. Configure the plugin

Edit `meshroom/config.json` to point to your LINO_UniPS clone:

```json
[
    {
        "key": "LINO_UNIPS_PATH",
        "type": "path",
        "value": "/absolute/path/to/LINO_UniPS"
    }
]
```

### 5. Download weights

Download the pre-trained model weights (`.pth` file) and place them in the LINO_UniPS code directory or in a `weights/` subdirectory. The plugin will auto-detect `.pth` files.

Alternatively, set the `LINO_MODEL_URL` environment variable and the model will be downloaded automatically from HuggingFace on first run.

### 6. Register the plugin in Meshroom

Set the `MESHROOM_PLUGINS_PATH` environment variable:

```bash
# Linux
export MESHROOM_PLUGINS_PATH=/path/to/your/plugins/mrLINOUniPS:$MESHROOM_PLUGINS_PATH

# Windows
set MESHROOM_PLUGINS_PATH=C:\path\to\mrLINOUniPS;%MESHROOM_PLUGINS_PATH%
```

Launch Meshroom: the **LINOUniPS** node appears under the **Photometric Stereo** category.

---

## Plugin Structure

```
mrLINOUniPS/
├── meshroom/
│   ├── config.json                # Plugin configuration (LINO_UNIPS_PATH)
│   └── LINOUniPS/
│       ├── __init__.py
│       └── LINOUniPS.py           # Meshroom node definition
├── venv/                          # Python virtual environment (or symlink, see step 3)
├── requirements.txt               # Python dependencies
└── README.md
```

For more details on how Meshroom plugins work, see:
- [Meshroom Plugin Install Guide](https://github.com/alicevision/Meshroom/blob/develop/INSTALL_PLUGINS.md)
- [mrHelloWorld](https://github.com/meshroomHub/mrHelloWorld): step-by-step tutorials for building Meshroom plugins

---

## Node Parameters

### Inputs

| Parameter | Label | Description |
|-----------|-------|-------------|
| `inputSfm` | Input SfMData | SfMData JSON file with multi-lighting views grouped by poseId **(required)** |
| `maskFolder` | Mask Folder | Folder with mask PNGs named by poseId or viewId |
| `downscale` | Downscale Factor | Integer downscale factor for input images (1-8, default: 1) |
| `nbImages` | Number of Images | Number of lighting images per pose (-1 = all) |
| `taskName` | Task Name | Model task configuration: `Real` or `DiLiGenT` |
| `useGpu` | Use GPU | Use GPU for inference (default: true) |
| `linoUniPsPath` | LINO_UniPS Path | Path to LINO_UniPS code (set via `config.json`) |

### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Folder containing normal map PNGs |
| `outputJson` | JSON file mapping poseIds to normal map paths |

---

## Acknowledgements

This work is supported by [**DOPAMIn**](https://www.cnrsinnovation.com/actualite/une-seconde-promotion-pour-le-programme-open-7-nouveaux-logiciels-scientifiques-a-valoriser/) (*Diffusion Open de Photogrammetrie par AliceVision/Meshroom pour l'Industrie*), selected in the 2024 cohort of the [**OPEN**](https://www.cnrsinnovation.com/open/) programme run by [CNRS Innovation](https://www.cnrsinnovation.com/). OPEN supports the valorization of open-source scientific software by providing dedicated developer resources, governance expertise, and industry partnership support.

**Lead researcher:** [Jean-Denis Durou](https://cv.hal.science/jean-denis-durou), [IRIT](https://www.irit.fr/) (INP-Toulouse)

---

## Related Projects

| Project | Description |
|---------|-------------|
| [LINO_UniPS](https://github.com/houyuanchen111/LINO_UniPS) | Light-invariant normal estimator for universal photometric stereo |
| [mrSDMUniPS](https://github.com/meshroomHub/mrSDMUniPS) | Meshroom plugin for SDM-UniPS photometric stereo |
| [mrOpenRNb](https://github.com/meshroomHub/mrOpenRNb) | Meshroom plugin for neural surface reconstruction from normals |

---

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
