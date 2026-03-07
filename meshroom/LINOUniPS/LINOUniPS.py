__version__ = "1.0"

import json
import os
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class LINOUniPS(desc.Node):
    """
    Multi-view photometric stereo normal estimation using LINO_UniPS.

    Reads an SfMData JSON with multi-lighting views grouped by poseId,
    runs normal estimation per pose, and outputs normal maps with an
    output JSON referencing all results.
    """

    category = "Photometric Stereo"
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize("inputSfm")

    documentation = """
    Estimate surface normals from multi-lighting images using LINO_UniPS.

    **Inputs:**
    - SfMData JSON with views grouped by poseId (multi-lighting)
    - Optional mask folder (masks named by poseId or viewId)

    **Processing:**
    - Images are cropped around masks internally for efficiency
    - Normal maps are uncropped back to full resolution transparently

    **Outputs:**
    - Normal map PNGs (16-bit) per pose
    - JSON file mapping poseIds to normal map paths
    """

    inputs = [
        desc.File(
            name="inputSfm",
            label="Input SfMData",
            description="SfMData JSON file with multi-lighting views "
                        "grouped by poseId.",
            value="",
        ),
        desc.File(
            name="maskFolder",
            label="Mask Folder",
            description="Folder with mask PNGs named by poseId or viewId "
                        "(e.g. '12345.png'). Optional.",
            value="",
        ),
        desc.IntParam(
            name="downscale",
            label="Downscale Factor",
            description="Integer downscale factor for input images "
                        "(1 = original, 2 = half, 4 = quarter).",
            value=1,
            range=(1, 8, 1),
        ),
        desc.IntParam(
            name="nbImages",
            label="Number of Images",
            description="Number of lighting images per pose to use "
                        "(-1 = all).",
            value=-1,
            range=(-1, 200, 1),
        ),
        desc.BoolParam(
            name="useGpu",
            label="Use GPU",
            description="Use GPU for inference.",
            value=True,
            invalidate=False,
        ),
        desc.File(
            name="linoUniPsPath",
            label="LINO_UniPS Path",
            description="Path to LINO_UniPS code directory. "
                        "Set via config.json key LINO_UNIPS_PATH.",
            value="${LINO_UNIPS_PATH}",
            advanced=True,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level for logging.",
            values=VERBOSE_LEVEL,
            value="info",
            exclusive=True,
        ),
    ]

    outputs = [
        desc.File(
            name="outputFolder",
            label="Output Folder",
            description="Folder containing normal map PNGs.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="outputSfmDataNormal",
            label="SfMData Normal",
            description="Output SfMData file referencing normal maps.",
            value="{nodeCacheFolder}/normalMaps.sfm",
        ),
        desc.File(
            name="outputMaskFolder",
            label="Mask Folder",
            description="Folder with masks extracted from alpha channels.",
            value="{nodeCacheFolder}/masks",
            group="",
        ),
    ]

    @staticmethod
    def _scale_intrinsics(sfm, downscale):
        """Scale intrinsics and view dimensions to match downscaled images."""
        if downscale <= 1:
            return
        f = float(downscale)
        for intr in sfm.get("intrinsics", []):
            for key in ("width", "height"):
                if key in intr:
                    intr[key] = str(int(int(float(str(intr[key]))) / f))
            if "principalPoint" in intr:
                pp = intr["principalPoint"]
                intr["principalPoint"] = [
                    str(float(str(pp[0])) / f),
                    str(float(str(pp[1])) / f),
                ]
            if "pxFocalLength" in intr:
                pfl = intr["pxFocalLength"]
                if isinstance(pfl, list):
                    intr["pxFocalLength"] = [pfl[0] / f, pfl[1] / f]
                else:
                    intr["pxFocalLength"] = float(pfl) / f
        for view in sfm.get("views", []):
            for key in ("width", "height"):
                if key in view:
                    view[key] = str(int(int(float(str(view[key]))) / f))

    def _create_output_sfm(self, sfm_data, output_folder,
                           map_type, suffix, logger, downscale=1):
        """Create an output SfMData JSON that references generated map files."""
        import copy
        sfm = copy.deepcopy(sfm_data)

        views = sfm.get("views", [])
        representative_views = []
        for view in views:
            view_id = str(view.get("viewId", ""))
            pose_id = str(view.get("poseId", ""))
            if view_id == pose_id:
                map_path = os.path.join(output_folder,
                                        "{}{}".format(pose_id, suffix))
                view["path"] = map_path
                representative_views.append(view)

        sfm["views"] = representative_views
        self._scale_intrinsics(sfm, downscale)

        output_path = os.path.join(output_folder, "{}.sfm".format(map_type))
        with open(output_path, "w") as f:
            json.dump(sfm, f, indent=4)

        logger.info("Saved {} to {}".format(map_type, output_path))
        return output_path

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            # Validate inputs
            input_sfm = chunk.node.inputSfm.value
            if not input_sfm:
                raise RuntimeError("inputSfm is required but empty.")
            if not os.path.exists(input_sfm):
                raise RuntimeError(
                    "Input SfM file not found: {}".format(input_sfm))

            mask_folder = chunk.node.maskFolder.value or ""
            if mask_folder and not os.path.isdir(mask_folder):
                chunk.logger.warning(
                    "Mask folder not found, continuing without masks: {}".format(mask_folder))
                mask_folder = ""

            # Resolve LINO_UniPS path
            lino_path = chunk.node.linoUniPsPath.evalValue
            if not lino_path or not os.path.isdir(lino_path):
                raise RuntimeError(
                    "LINO_UNIPS_PATH is empty or not a valid directory. "
                    "Set it in config.json. Got: '{}'".format(lino_path))

            # Import LINO_UniPS modules upfront, then restore sys.path
            import sys
            original_path = sys.path[:]
            sys.path.insert(0, lino_path)
            try:
                from inference_sfm import run_sfm_inference, load_sfm
                # Pre-import hubconf and its dependencies so they are
                # cached in sys.modules (run_sfm_inference imports them lazily)
                import hubconf
                import src.models
                import src.data
            except ImportError as e:
                raise RuntimeError(
                    "Failed to import from LINO_UniPS at {}: {}".format(
                        lino_path, e))
            finally:
                sys.path[:] = original_path

            # Device selection
            import torch
            use_gpu = chunk.node.useGpu.value
            use_cuda = use_gpu and torch.cuda.is_available()
            if use_gpu and not use_cuda:
                chunk.logger.warning("CUDA not available, falling back to CPU")

            # Output folder
            output_folder = chunk.node.outputFolder.value
            os.makedirs(output_folder, exist_ok=True)

            # Find weights file
            weights_path = None
            for candidate in ["lino_unips.pth", "lino.pth", "model.pth"]:
                p = os.path.join(lino_path, candidate)
                if os.path.isfile(p):
                    weights_path = p
                    break
            if not weights_path:
                # Search in weights/ subdirectory
                weights_dir = os.path.join(lino_path, "weights")
                if os.path.isdir(weights_dir):
                    for f in os.listdir(weights_dir):
                        if f.endswith(".pth"):
                            weights_path = os.path.join(weights_dir, f)
                            break
            if not weights_path:
                # Search in torch hub cache
                torch_cache = os.path.join(
                    os.path.expanduser("~"),
                    ".cache", "torch", "hub", "checkpoints", "lino.pth")
                if os.path.isfile(torch_cache):
                    weights_path = torch_cache

            # Run inference
            chunk.logger.info("Starting LINO_UniPS inference...")
            chunk.logger.info("  Input SfM: {}".format(input_sfm))
            chunk.logger.info("  Masks: {}".format(mask_folder or "(none)"))
            chunk.logger.info("  Downscale: {}".format(
                chunk.node.downscale.value))
            chunk.logger.info("  GPU: {}".format(use_cuda))
            if weights_path:
                chunk.logger.info("  Weights: {}".format(weights_path))
            else:
                chunk.logger.info("  Weights: (will download from HuggingFace)")

            run_sfm_inference(
                sfm_path=input_sfm,
                output_folder=output_folder,
                mask_folder=mask_folder if mask_folder else None,
                mask_output_folder=chunk.node.outputMaskFolder.value,
                nb_img=chunk.node.nbImages.value,
                downscale=chunk.node.downscale.value,
                use_cuda=use_cuda,
                task_name="Real",
                weights_path=weights_path,
            )

            chunk.logger.info("Inference done.")

            # Load SfM data for output SfMData generation
            sfm_data = load_sfm(input_sfm)

            # Create output SfMData for normal maps
            self._create_output_sfm(
                sfm_data, output_folder,
                "normalMaps", ".png", chunk.logger,
                downscale=chunk.node.downscale.value)

        finally:
            # GPU cleanup
            try:
                import gc
                import torch as _torch
                gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            chunk.logManager.end()
