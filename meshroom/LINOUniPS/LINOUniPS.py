__version__ = "2.0"

import os

from meshroom.core import desc

from . import psCommon


class LINOUniPS(desc.Node):
    """Multi-view photometric stereo normal estimation with LINO-UniPS."""

    category = "Photometric Stereo"
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize("inputSfm")

    documentation = """
Estimate one normal map per multi-lighting pose with LINO-UniPS (universal photometric stereo: unknown lighting).

**Inputs:** an SfMData where the lighting images of a pose share the same poseId (as created by CameraInit for
multi-lighting folders), e.g. the undistorted images of ExportImages. Poses with fewer than 'Min Views Per Pose'
views (photogrammetry images) are ignored, so a mixed multi-view / multi-light SfMData can be used as is.

**Masks:** from a mask folder (<poseId>.png or <viewId>.png) or from the alpha channel of the images; the
per-image masks of a pose are combined by vote ('Mask Vote Threshold').

**Outputs:** one normal map per pose (<poseId>.png|exr, OpenGL camera frame by default), the pose masks
(masks/<poseId>.png: pixels with a normal) and an SfMData referencing the normal maps (one view per pose, with the
intrinsics scaled by the downscale factor), ready for RNb-NeuS2.

The data handling (SfMData, image selection, masks, outputs) is common to the LINOUniPS, UniMSPS and SDMUniPS
nodes; see the advanced options.
"""

    inputs = psCommon.inputAttributes() + [
        desc.IntParam(
            name="cropMargin",
            label="Crop Margin",
            description="Margin (pixels) around the bounding box of the mask. The whole image is processed when the "
                        "box is closer than this margin to the image border.",
            value=8,
            range=(0, 256, 1),
            advanced=True,
        ),
        desc.IntParam(
            name="maxProcessingSize",
            label="Max Processing Size",
            description="Maximum side of the square network input: the crop is resized to "
                        "max(512, min(Max Processing Size, floor(crop side / 512) * 512)).",
            value=6000,
            range=(512, 8192, 512),
            advanced=True,
        ),
        desc.ChoiceParam(
            name="outputInterpolation",
            label="Output Interpolation",
            description="Resampling of the network prediction back to the crop size.",
            value="cubic",
            values=["area", "linear", "cubic"],
            exclusive=True,
            advanced=True,
        ),
        desc.File(
            name="modelPath",
            label="Model",
            description="LINO-UniPS weights (.pth). If empty: <plugin>/weights/lino.pth, then "
                        "<LINO_UniPS>/weights/lino.pth, then ~/.cache/torch/hub/checkpoints/lino.pth.",
            value="",
            advanced=True,
        ),
        desc.File(
            name="linoUniPsPath",
            label="LINO_UniPS Path",
            description="LINO_UniPS code directory, used if the package is not installed in the plugin environment.",
            value="${LINO_UNIPS_PATH}",
            advanced=True,
            invalidate=False,
        ),
    ] + psCommon.advancedInputAttributes() + psCommon.settingsAttributes()

    outputs = psCommon.outputAttributes()

    @staticmethod
    def findWeights(node):
        if node.modelPath.value:
            return node.modelPath.value
        candidates = [os.path.join(os.path.dirname(__file__), "..", "..", "weights", "lino.pth")]
        if node.linoUniPsPath.evalValue:
            candidates.append(os.path.join(node.linoUniPsPath.evalValue, "weights", "lino.pth"))
        candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "lino.pth"))
        for path in candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)
        raise RuntimeError("LINO-UniPS weights not found, set 'Model' or download them (download_weights.sh). "
                           "Searched: {}".format(", ".join(candidates)))

    @staticmethod
    def importApi(node):
        try:
            import meshroom_predict
        except ImportError:
            import sys
            path = node.linoUniPsPath.evalValue
            if not path or not os.path.isdir(path):
                raise RuntimeError("LINO_UniPS is not installed in the plugin environment and 'LINO_UniPS Path' is "
                                   "invalid: '{}'".format(path))
            sys.path.insert(0, path)
            import meshroom_predict
        return meshroom_predict

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            import torch
            node = chunk.node
            api = self.importApi(node)
            weights = self.findWeights(node)
            chunk.logger.info("LINO-UniPS weights: {}".format(weights))
            predictor = api.loadModel(weights, useGpu=node.useGpu.value, logger=chunk.logger)

            def predict(images, mask):
                return api.predict(predictor, images, mask, cropMargin=node.cropMargin.value,
                                   maxProcessingSize=node.maxProcessingSize.value,
                                   outputInterpolation=node.outputInterpolation.value)

            def cleanup():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            psCommon.processPoses(chunk, predict, cleanup=cleanup)
        finally:
            chunk.logManager.end()
