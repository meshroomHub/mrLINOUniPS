__version__ = "2.1"
__author__ = "Baptiste Brument"
__license__ = "MPL-2.0"

import os

from meshroom.core import desc

from . import psCommon

# Networks: "normal" (LINO-UniPS, normals only) and "pbr" (LINO-UniPS PBR: normals, base color, roughness and
# metallic from the same features), with their default weights file
WEIGHTS_FILES = {"normal": "lino.pth", "pbr": "lino_pbr.pth"}
EXTRA_MAPS = ("albedo",)


def requiredModels(normalModel, estimateAlbedo):
    """Networks run on each pose: the normal model, plus the PBR model for the albedo when it is not the normal
    model."""
    models = [normalModel]
    if estimateAlbedo and normalModel != "pbr":
        models.append("pbr")
    return models


def uidIgnoredWhen(value, attribute):
    """Attribute whose given value does not contribute to the node UID: an option added with its default value keeps
    the UID (and the computed caches) of the existing nodes. desc.BoolParam and desc.ChoiceParam have no
    'uidIgnoreValue' argument (only desc.StringParam has one): it is set on the description."""
    attribute._uidIgnoreValue = value
    return attribute


def usesModel(node, model):
    return model in requiredModels(node.normalModel.value, node.estimateAlbedo.value)


def makePredict(api, predictors, normalModel, estimateAlbedo, **options):
    """psCommon predict(images, mask) function: normals of the normal model and, if estimateAlbedo, the albedo
    (base color) of the PBR model, from the same network pass when the PBR model also gives the normals.

    Args:
        api: the meshroom_predict module of LINO_UniPS.
        predictors: {model: predictor} for the models of requiredModels().
        options: options of api.predict (cropMargin, maxProcessingSize, outputInterpolation).
    """
    def predict(images, mask):
        maps = api.predict(predictors[normalModel], images, mask, **options)
        result = {"normal": maps["normal"]}
        if estimateAlbedo:
            pbrMaps = maps if normalModel == "pbr" else api.predict(predictors["pbr"], images, mask, **options)
            result["albedo"] = pbrMaps["albedo"]
        return result
    return predict


class LINOUniPS(desc.Node):
    """Multi-view photometric stereo normal (and albedo) estimation with LINO-UniPS."""

    # Credits of the wrapped method (shown in the node info, with the plugin author and license)
    __nodeInfo__ = [
        ("method", "LINO-UniPS: Light of Normals, Unified Feature Representation for Universal Photometric Stereo "
                   "(H. Li, H. Chen, C. Ye et al., arXiv:2506.18882, 2025)"),
        ("methodLicense", "MIT"),
        ("methodRepository", "https://github.com/houyuanchen111/LINO_UniPS"),
    ]

    category = "Photometric Stereo"
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize("inputSfm")

    documentation = """
Estimate one normal map per multi-lighting pose with LINO-UniPS (universal photometric stereo: unknown lighting),
and optionally one albedo map per pose with the LINO-UniPS PBR network.

**Inputs:** an SfMData where the lighting images of a pose share the same poseId (as created by CameraInit for
multi-lighting folders), e.g. the undistorted images of ExportImages. Poses with fewer than 'Min Views Per Pose'
views (photogrammetry images) are ignored, so a mixed multi-view / multi-light SfMData can be used as is.

**Masks:** from a mask folder (<poseId>.png or <viewId>.png) or from the alpha channel of the images; the
per-image masks of a pose are combined by vote ('Mask Vote Threshold').

**Outputs:** one normal map per pose (<poseId>.png|exr, OpenGL camera frame by default), the pose masks
(masks/<poseId>.png: pixels with a normal) and an SfMData referencing the normal maps (one view per pose, with the
intrinsics scaled by the downscale factor), ready for RNb-NeuS2.

**Albedo ('Estimate Albedo'):** the base color predicted by the LINO-UniPS PBR network (lino_pbr.pth), in [0, 1],
zero outside the pose mask: albedo/<poseId>.png|exr (same size and format as the normal maps) and an SfMData
(albedoMaps.sfm, same views and intrinsics as normalMaps.sfm) for the albedo input of RNb-NeuS2. 'Normal Model'
chooses the network of the normals: 'pbr' gives the normals and the albedo in one network pass, 'normal' keeps
the normals of the LINO-UniPS normal model and runs the PBR network as well (two passes per pose).

The data handling (SfMData, image selection, masks, outputs) is common to the LINOUniPS, UniMSPS and SDMUniPS
nodes; see the advanced options.
"""

    inputs = psCommon.inputAttributes() + [
        uidIgnoredWhen(False, desc.BoolParam(
            name="estimateAlbedo",
            label="Estimate Albedo",
            description="Also estimate one albedo map per pose: the base color predicted by the LINO-UniPS PBR "
                        "network (lino_pbr.pth), written next to the normal maps (albedo/) with its SfMData, for the "
                        "albedo input of RNb-NeuS2. Without 'Normal Model' = pbr, the PBR network runs in addition "
                        "to the normal model (two network passes per pose).",
            value=False,
        )),
        uidIgnoredWhen("normal", desc.ChoiceParam(
            name="normalModel",
            label="Normal Model",
            description="Network predicting the normals:\n"
                        " - normal: LINO-UniPS (lino.pth), the published normal estimation model.\n"
                        " - pbr: LINO-UniPS PBR (lino_pbr.pth), trained to predict the normals along with the base "
                        "color, roughness and metallic: normals and albedo in one network pass.",
            value="normal",
            values=list(WEIGHTS_FILES),
            exclusive=True,
        )),
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
            enabled=lambda node: usesModel(node, "normal"),
        ),
        desc.File(
            name="pbrModelPath",
            label="PBR Model",
            description="LINO-UniPS PBR weights (.pth), used for the albedo or the 'pbr' normal model. If empty: "
                        "<plugin>/weights/lino_pbr.pth, then <LINO_UniPS>/weights/lino_pbr.pth, then "
                        "~/.cache/torch/hub/checkpoints/lino_pbr.pth (download_weights.sh pbr).",
            value="",
            advanced=True,
            enabled=lambda node: usesModel(node, "pbr"),
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

    outputs = psCommon.outputAttributes(extraMaps=EXTRA_MAPS, extraEnabled=lambda node: node.estimateAlbedo.value)

    @staticmethod
    def findWeights(node, model="normal"):
        """Weights of a network ("normal" or "pbr"): the 'Model' / 'PBR Model' input if set, otherwise the first
        existing default file (plugin weights/, LINO_UniPS weights/, torch hub cache)."""
        attribute = node.pbrModelPath if model == "pbr" else node.modelPath
        if attribute.value:
            return attribute.value
        fileName = WEIGHTS_FILES[model]
        candidates = [os.path.join(os.path.dirname(__file__), "..", "..", "weights", fileName)]
        if node.linoUniPsPath.evalValue:
            candidates.append(os.path.join(node.linoUniPsPath.evalValue, "weights", fileName))
        candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", fileName))
        for path in candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)
        raise RuntimeError("LINO-UniPS {} weights not found, set '{}' or download them (download_weights.sh{}). "
                           "Searched: {}".format(model, "PBR Model" if model == "pbr" else "Model",
                                                 " pbr" if model == "pbr" else "", ", ".join(candidates)))

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
            normalModel, estimateAlbedo = node.normalModel.value, node.estimateAlbedo.value
            predictors = {}
            for model in requiredModels(normalModel, estimateAlbedo):
                weights = self.findWeights(node, model)
                chunk.logger.info("LINO-UniPS {} weights: {}".format(model, weights))
                # the model is only passed for the PBR network: the default keeps working with a LINO_UniPS
                # predict API older than the PBR support
                options = {"model": model} if model != "normal" else {}
                predictors[model] = api.loadModel(weights, useGpu=node.useGpu.value, logger=chunk.logger, **options)
            predict = makePredict(api, predictors, normalModel, estimateAlbedo, cropMargin=node.cropMargin.value,
                                  maxProcessingSize=node.maxProcessingSize.value,
                                  outputInterpolation=node.outputInterpolation.value)

            def cleanup():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            psCommon.processPoses(chunk, predict, extraMaps=EXTRA_MAPS if estimateAlbedo else (), cleanup=cleanup)
        finally:
            chunk.logManager.end()
