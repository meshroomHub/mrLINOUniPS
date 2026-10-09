"""Common input/output layer of the universal photometric stereo plugins.

Shared, as an identical copy, by mrLINOUniPS, mrUniMSPS and mrSDMUniPS (keep the copies in sync, see
PS_COMMON_VERSION). It implements everything but the network inference, so that the three nodes handle
their data in exactly the same way:

- the node attributes shared by the three nodes (same names, defaults and semantics),
- SfMData loading, grouping of the views by pose and filtering of the multi-lighting poses,
- selection and loading of the lighting images (8/16-bit and float images, downscale),
- pose masks (mask files or alpha channels, binarization, border cleaning, vote between images),
- writing of the normal maps (and of any other map), of the pose masks and of the output SfMData,
- per-pose seeding, error handling and failure policy.

A method plugin only provides a ``predict(images, mask)`` function: it receives the lighting images of one
pose (float32 RGB, H x W x 3) and the pose mask (bool, H x W) and returns a dict of maps at the same
resolution, with at least "normal": unit normals in the OpenGL camera frame (x right, y up, z towards the
camera), zero where undefined.
"""

PS_COMMON_VERSION = "1.0"

import copy
import json
import math
import os
import time
import traceback

# Must be set before OpenCV is imported to read/write EXR files
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


# --------------------------------------------------------------------------------------------------------
# Node attributes
# --------------------------------------------------------------------------------------------------------

def inputAttributes():
    """Basic inputs shared by the photometric stereo nodes."""
    return [
        desc.File(
            name="inputSfm",
            label="SfMData",
            description="SfMData with the multi-lighting views (e.g. undistorted images from ExportImages). "
                        "Views sharing the same poseId are the lighting images of one pose; poses with fewer than "
                        "'Min Views Per Pose' views (e.g. photogrammetry images) are ignored.",
            value="",
        ),
        desc.File(
            name="maskFolder",
            label="Mask Folder",
            description="Optional folder of masks named <poseId>.png (one mask per pose) or <viewId>.png (one mask "
                        "per image, combined by vote). A non-opaque RGBA mask file uses its alpha channel, "
                        "e.g. an ambient image exported with a SAM3 mask. Without mask files, masks are "
                        "extracted from the alpha channel of the input images.",
            value="",
        ),
        desc.IntParam(
            name="downscale",
            label="Downscale Factor",
            description="Integer downscale factor applied to the input images (and to the output maps and "
                        "intrinsics).",
            value=1,
            range=(1, 8, 1),
        ),
        desc.IntParam(
            name="nbImages",
            label="Number Of Images",
            description="Maximum number of lighting images used per pose (-1: all images). "
                        "The GPU memory used by the network grows with this number.",
            value=-1,
            range=(-1, 200, 1),
        ),
    ]


def advancedInputAttributes():
    """Advanced inputs shared by the photometric stereo nodes: every heuristic of the data handling."""
    return [
        desc.IntParam(
            name="minViewsPerPose",
            label="Min Views Per Pose",
            description="Minimum number of views sharing a poseId for the pose to be processed as a "
                        "multi-lighting pose.",
            value=3,
            range=(1, 100, 1),
            advanced=True,
        ),
        desc.ChoiceParam(
            name="imageSelection",
            label="Image Selection",
            description="How the lighting images are chosen when 'Number Of Images' is lower than the number of "
                        "images of a pose (images sorted by path):\n"
                        " - random: random subset (reproducible, see 'Seed').\n"
                        " - uniform: evenly spaced images.\n"
                        " - first: first images.",
            value="random",
            values=["random", "uniform", "first"],
            exclusive=True,
            advanced=True,
        ),
        desc.IntParam(
            name="seed",
            label="Seed",
            description="Seed of the random image selection and of the network randomness. "
                        "It is combined with the poseId, so that each pose is reproducible on its own.",
            value=42,
            range=(0, 100000, 1),
            advanced=True,
        ),
        desc.BoolParam(
            name="linearizeInput",
            label="Linearize Input",
            description="Convert 8/16-bit images from sRGB to linear values before the network "
                        "(float images are assumed to be linear already).",
            value=False,
            advanced=True,
        ),
        desc.FloatParam(
            name="maskThreshold",
            label="Mask Threshold",
            description="Binarization threshold of the masks (mask files and alpha channels), as a fraction of "
                        "the value range (0.5: 127 for 8-bit, 32767 for 16-bit, 0.5 for float).",
            value=0.5,
            range=(0.0, 1.0, 0.01),
            advanced=True,
        ),
        desc.FloatParam(
            name="maskVoteThreshold",
            label="Mask Vote Threshold",
            description="Combination of the per-image masks of a pose: a pixel belongs to the pose mask when the "
                        "fraction of masks containing it is greater than this threshold.\n"
                        " - 0.5: strict majority (robust to a few failed masks).\n"
                        " - 1.0: present in every mask (intersection).\n"
                        " - 0.0: present in at least one mask (union).",
            value=0.5,
            range=(0.0, 1.0, 0.01),
            advanced=True,
        ),
        desc.BoolParam(
            name="maskRemoveBorderComponents",
            label="Remove Border Components",
            description="For masks extracted from alpha channels: remove the connected components touching the "
                        "image border (e.g. the valid area of undistorted images). Disable it when the object "
                        "itself touches the image border.",
            value=True,
            advanced=True,
        ),
        desc.BoolParam(
            name="maskUseGlobalFile",
            label="Use Global Mask File",
            description="Use <Mask Folder>/mask.png for every pose without a specific mask file.",
            value=False,
            advanced=True,
        ),
        desc.ChoiceParam(
            name="normalConvention",
            label="Normal Convention",
            description="Camera frame of the output normals:\n"
                        " - opengl: x right, y up, z towards the camera (expected by RNb-NeuS2).\n"
                        " - opencv: x right, y down, z forward (AliceVision camera frame).",
            value="opengl",
            values=["opengl", "opencv"],
            exclusive=True,
            advanced=True,
        ),
        desc.BoolParam(
            name="keepLandmarks",
            label="Keep Landmarks",
            description="Keep the 3D landmark positions of the input SfMData in the output SfMData, without their "
                        "observations (used by RNb-NeuS2 to normalize the scene when no mask is available).",
            value=True,
            advanced=True,
        ),
        desc.ChoiceParam(
            name="failurePolicy",
            label="Failure Policy",
            description="When the node fails because of poses that could not be processed (e.g. out of memory):\n"
                        " - noPose: only if no pose could be processed.\n"
                        " - anyPose: as soon as one pose could not be processed.\n"
                        " - never: never (the failed poses are only reported in the log).",
            value="noPose",
            values=["noPose", "anyPose", "never"],
            exclusive=True,
            advanced=True,
        ),
    ]


def settingsAttributes():
    """Output format, device and verbosity, shared by the photometric stereo nodes."""
    return [
        desc.ChoiceParam(
            name="outputFormat",
            label="Output Format",
            description="Format of the output maps:\n"
                        " - png16: 16-bit PNG, value = (n + 1) / 2 * 65535 (background: 32768, i.e. a zero vector).\n"
                        " - exr: float32 EXR with the raw values (background: 0).",
            value="png16",
            values=["png16", "exr"],
            exclusive=True,
        ),
        desc.BoolParam(
            name="useGpu",
            label="Use GPU",
            description="Use the GPU for the inference (the CPU is used when no GPU is available).",
            value=True,
            invalidate=False,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug).",
            value="info",
            values=VERBOSE_LEVEL,
            exclusive=True,
        ),
    ]


def mapExtension(node):
    return ".exr" if node.outputFormat.value == "exr" else ".png"


def outputAttributes(extraMaps=(), normalEnabled=True, extraEnabled=True):
    """Outputs shared by the photometric stereo nodes, plus one SfMData and one image output per extra map.

    normalEnabled / extraEnabled: bool or function(node) -> bool, to disable the outputs of the maps a node
    does not compute with its current settings.
    """
    outputs = [
        desc.File(
            name="outputFolder",
            label="Output Folder",
            description="Folder containing the normal maps (<poseId>.png|exr), the pose masks (masks/) and any "
                        "other map (<map>/).",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="outputSfmDataNormal",
            label="Normal Maps SfMData",
            description="SfMData referencing the normal maps (one view per pose).",
            value="{nodeCacheFolder}/normalMaps.sfm",
            enabled=normalEnabled,
        ),
        desc.File(
            name="normalMaps",
            label="Normal Maps",
            description="Normal maps.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/<VIEW_ID>" + mapExtension(attr.node),
            commandLineGroup="",
            enabled=normalEnabled,
        ),
        desc.File(
            name="outputMaskFolder",
            label="Mask Folder",
            description="Folder with the pose masks (<poseId>.png, 0/255): the pixels where a normal is defined.",
            value="{nodeCacheFolder}/masks",
        ),
        desc.File(
            name="masks",
            label="Masks",
            description="Pose masks.",
            semantic="image",
            value="{nodeCacheFolder}/masks/<VIEW_ID>.png",
            commandLineGroup="",
        ),
    ]
    for mapName in extraMaps:
        label = mapName.capitalize()
        outputs.append(desc.File(
            name="outputSfmData" + label,
            label=label + " Maps SfMData",
            description="SfMData referencing the {} maps (one view per pose).".format(mapName),
            value="{{nodeCacheFolder}}/{}Maps.sfm".format(mapName),
            enabled=extraEnabled,
        ))
        outputs.append(desc.File(
            name=mapName + "Maps",
            label=label + " Maps",
            description=label + " maps.",
            semantic="image",
            value=(lambda name: lambda attr: "{nodeCacheFolder}/" + name + "/<VIEW_ID>" + mapExtension(attr.node))(mapName),
            commandLineGroup="",
            enabled=extraEnabled,
        ))
    return outputs


class Params:
    """Values of the common attributes of a node."""

    def __init__(self, node):
        self.inputSfm = node.inputSfm.value
        self.maskFolder = node.maskFolder.value or ""
        self.downscale = int(node.downscale.value)
        self.nbImages = int(node.nbImages.value)
        self.minViewsPerPose = int(node.minViewsPerPose.value)
        self.imageSelection = node.imageSelection.value
        self.seed = int(node.seed.value)
        self.linearizeInput = bool(node.linearizeInput.value)
        self.maskThreshold = float(node.maskThreshold.value)
        self.maskVoteThreshold = float(node.maskVoteThreshold.value)
        self.maskRemoveBorderComponents = bool(node.maskRemoveBorderComponents.value)
        self.maskUseGlobalFile = bool(node.maskUseGlobalFile.value)
        self.normalConvention = node.normalConvention.value
        self.keepLandmarks = bool(node.keepLandmarks.value)
        self.failurePolicy = node.failurePolicy.value
        self.outputFormat = node.outputFormat.value
        self.useGpu = bool(node.useGpu.value)
        self.outputFolder = node.outputFolder.value
        self.outputMaskFolder = node.outputMaskFolder.value

    def describe(self):
        return ", ".join("{}={}".format(k, v) for k, v in sorted(vars(self).items()))


# --------------------------------------------------------------------------------------------------------
# SfMData
# --------------------------------------------------------------------------------------------------------

def loadSfm(path, tmpFolder):
    """Load an SfMData file as a JSON dict.

    JSON files (.sfm, .json) are read directly; any other format readable by AliceVision (.abc, .usda...) is
    converted to JSON through pyalicevision (the temporary file is written in tmpFolder).
    """
    if not path or not os.path.isfile(path):
        raise RuntimeError("Input SfMData not found: '{}'".format(path))
    if os.path.splitext(path)[1].lower() in (".sfm", ".json"):
        with open(path) as f:
            return json.load(f)
    try:
        from pyalicevision import sfmData as avSfmData
        from pyalicevision import sfmDataIO
    except ImportError:
        with open(path) as f:
            return json.load(f)
    data = avSfmData.SfMData()
    if not sfmDataIO.load(data, path, sfmDataIO.ALL):
        raise RuntimeError("Cannot load the SfMData: '{}'".format(path))
    os.makedirs(tmpFolder, exist_ok=True)
    tmpPath = os.path.join(tmpFolder, "inputSfmData.sfm")
    if not sfmDataIO.save(data, tmpPath, sfmDataIO.ALL):
        raise RuntimeError("Cannot convert the SfMData to JSON: '{}'".format(path))
    try:
        with open(tmpPath) as f:
            return json.load(f)
    finally:
        os.remove(tmpPath)


def saveSfm(sfm, path):
    with open(path, "w") as f:
        json.dump(sfm, f, indent=4)


def groupViewsByPose(views):
    """Group views by poseId (views of a pose sorted by image path, poses in order of first appearance)."""
    groups = {}
    for view in views:
        groups.setdefault(str(view.get("poseId", view.get("viewId"))), []).append(view)
    return {poseId: sorted(group, key=lambda v: v.get("path", "")) for poseId, group in groups.items()}


def filterPoses(sfm, minViewsPerPose, logger):
    """Keep the localized poses with at least minViewsPerPose views.

    Returns:
        (filtered SfMData dict, {poseId: views sorted by path})
    """
    groups = groupViewsByPose(sfm.get("views", []))
    localized = {str(p.get("poseId")) for p in sfm.get("poses", [])}
    kept = {poseId: views for poseId, views in groups.items()
            if len(views) >= minViewsPerPose and poseId in localized}
    nbFewViews = sum(1 for views in groups.values() if len(views) < minViewsPerPose)
    nbNotLocalized = sum(1 for poseId, views in groups.items()
                         if len(views) >= minViewsPerPose and poseId not in localized)
    if nbFewViews:
        logger.info("Ignoring {} pose(s) with fewer than {} views (not multi-lighting poses).".format(
            nbFewViews, minViewsPerPose))
    if nbNotLocalized:
        logger.warning("Ignoring {} multi-lighting pose(s) without a pose in the SfMData (not localized).".format(
            nbNotLocalized))
    if not kept:
        raise RuntimeError("No localized pose with at least {} views in the input SfMData: no multi-lighting "
                           "data.".format(minViewsPerPose))
    out = copy.deepcopy(sfm)
    out["views"] = [v for v in sfm.get("views", []) if str(v.get("poseId")) in kept]
    out["poses"] = [p for p in sfm.get("poses", []) if str(p.get("poseId")) in kept]
    return out, kept


def representativeView(poseId, views):
    """View holding the pose in the output SfMData: the view whose viewId is the poseId (as created by
    CameraInit for multi-lighting folders), None if there is none."""
    for view in views:
        if str(view.get("viewId")) == str(poseId):
            return view
    return None


def scaledPrincipalPoint(offset, size, downscale):
    """Principal point offset (from the image center, AliceVision JSON convention) after downscaling.

    AliceVision puts the center of pixel i at coordinate i, and the downscaled pixel i averages the input pixels
    [i * s, (i + 1) * s) with s = size / (size // d): its center is at (i + 0.5) * s - 0.5 in the input image.
    Hence pp' = (pp + 0.5) / s - 0.5 for the absolute principal point pp = offset + size / 2
    (pp / s would shift the image by 0.5 * (1 - 1 / s) pixel with respect to the camera model).
    """
    newSize = size // downscale
    scale = newSize / float(size)
    absolute = (offset + size / 2.0 + 0.5) * scale - 0.5
    return absolute - newSize / 2.0


def scaleIntrinsics(sfm, downscale):
    """Scale the intrinsics and the view dimensions to images downscaled by an integer factor
    (to width // d, height // d, see scaledPrincipalPoint).

    The focal length in mm is unchanged: the focal in pixels follows the width through the sensor width.
    """
    if downscale <= 1:
        return
    for intrinsic in sfm.get("intrinsics", []):
        width, height = int(float(str(intrinsic["width"]))), int(float(str(intrinsic["height"])))
        if "principalPoint" in intrinsic:
            offset = [float(str(v)) for v in intrinsic["principalPoint"]]
            intrinsic["principalPoint"] = [str(scaledPrincipalPoint(offset[0], width, downscale)),
                                           str(scaledPrincipalPoint(offset[1], height, downscale))]
        if "pxFocalLength" in intrinsic:  # legacy SfMData versions
            value = intrinsic["pxFocalLength"]
            if isinstance(value, list):
                intrinsic["pxFocalLength"] = [str(float(str(value[0])) * (width // downscale) / width),
                                              str(float(str(value[1])) * (height // downscale) / height)]
            else:
                intrinsic["pxFocalLength"] = str(float(str(value)) * (width // downscale) / width)
        intrinsic["width"], intrinsic["height"] = str(width // downscale), str(height // downscale)
    for view in sfm.get("views", []):
        for key in ("width", "height"):
            if key in view:
                view[key] = str(int(float(str(view[key]))) // downscale)


def buildOutputSfm(sfm, mapPaths, downscale, keepLandmarks):
    """SfMData referencing one map per pose, attached to the representative view of the pose.

    Args:
        sfm: filtered input SfMData (dict).
        mapPaths: {poseId: map path}.
    """
    out = copy.deepcopy(sfm)
    views = []
    for view in out.get("views", []):
        poseId = str(view.get("poseId"))
        if str(view.get("viewId")) == poseId and poseId in mapPaths:
            view["path"] = mapPaths[poseId]
            views.append(view)
    out["views"] = views
    out["poses"] = [p for p in out.get("poses", []) if str(p.get("poseId")) in mapPaths]
    if keepLandmarks:
        # the observations are full-resolution features of the input views: only the 3D positions are kept
        for landmark in out.get("structure", []):
            landmark["observations"] = []
    else:
        out.pop("structure", None)
    scaleIntrinsics(out, downscale)
    return out


# --------------------------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------------------------

def poseSeed(seed, poseId):
    """Seed of a pose: combines the node seed and the poseId (independent of the processing order)."""
    try:
        poseKey = int(poseId) % (2 ** 32)
    except ValueError:
        poseKey = sum(ord(c) * 31 ** i for i, c in enumerate(str(poseId))) % (2 ** 32)
    return int(np.random.SeedSequence([seed, poseKey]).generate_state(1)[0])


def selectViews(views, nbImages, mode, seed):
    """Choose at most nbImages views (all if nbImages <= 0), keeping their order."""
    n = len(views)
    if nbImages <= 0 or nbImages >= n:
        return list(views)
    if mode == "first":
        indices = range(nbImages)
    elif mode == "uniform":
        indices = np.round(np.linspace(0, n - 1, nbImages)).astype(int)
    elif mode == "random":
        indices = np.random.default_rng(seed).choice(n, nbImages, replace=False)
    else:
        raise ValueError("Unknown image selection mode: '{}'".format(mode))
    return [views[i] for i in sorted(set(int(i) for i in indices))]


def normalizeValues(array):
    """Integer image values to float32 in [0, 1] (float images are returned as float32, unchanged)."""
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32)
    raise ValueError("Unsupported image type: {}".format(array.dtype))


def readImage(path):
    """Read an image (8/16-bit or float, with or without alpha) without any color conversion.

    Returns:
        (rgb float32 H x W x 3, alpha float32 H x W or None, isFloat)
    """
    import cv2
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Cannot read the image: '{}'".format(path))
    isFloat = np.issubdtype(image.dtype, np.floating)
    image = normalizeValues(image)
    alpha = None
    if image.ndim == 2:
        rgb = np.repeat(image[:, :, None], 3, axis=2)
    elif image.shape[2] == 2:
        rgb, alpha = np.repeat(image[:, :, :1], 3, axis=2), image[:, :, 1]
    else:
        rgb = image[:, :, 2::-1]  # BGR(A) -> RGB
        if image.shape[2] >= 4:
            alpha = image[:, :, 3]
    return np.ascontiguousarray(rgb), alpha, isFloat


def srgbToLinear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def downscaleImage(image, downscale):
    """Downscale by an integer factor to (width // d, height // d), with area interpolation."""
    if downscale <= 1:
        return image
    import cv2
    h, w = image.shape[:2]
    return cv2.resize(image, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)


def loadImages(views, params):
    """Lighting images of a pose: float32 RGB, downscaled (and linearized if requested)."""
    images = []
    for view in views:
        rgb, _, isFloat = readImage(view["path"])
        if params.linearizeInput and not isFloat:
            rgb = srgbToLinear(rgb)
        images.append(downscaleImage(rgb, params.downscale))
    shapes = {im.shape for im in images}
    if len(shapes) != 1:
        raise RuntimeError("The images of a pose must have the same size, got {}".format(sorted(shapes)))
    return images


# --------------------------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------------------------

def binarizeMask(values, threshold):
    """Mask from integer or float values (first channel), threshold as a fraction of the value range."""
    values = normalizeValues(values)
    if values.ndim == 3:
        values = values[:, :, 0]
    return values > threshold


def removeBorderComponents(mask):
    """Remove the connected components (8-connectivity) touching the image border."""
    import cv2
    nbLabels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    cleaned = mask.copy()
    cleaned[np.isin(labels, border[border > 0])] = False
    return cleaned


def voteMasks(masks, threshold):
    """Combine binary masks: keep a pixel when the fraction of masks containing it is greater than threshold
    (threshold >= 1: present in every mask)."""
    if not masks:
        return None
    votes = np.zeros(masks[0].shape, np.uint16)
    for mask in masks:
        votes += mask
    if threshold >= 1.0:
        return votes == len(masks)
    return votes > threshold * len(masks)


def readMaskFile(path, threshold):
    import cv2
    values = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if values is None:
        raise RuntimeError("Cannot read the mask: '{}'".format(path))
    # ExportImages writes RGB(A) PNGs. When their alpha is not fully opaque,
    # it is the undistorted object mask; otherwise retain the first-channel
    # behavior for ordinary grayscale/RGB(A) mask files.
    if values.ndim == 3 and values.shape[2] == 4:
        opaque = np.iinfo(values.dtype).max if np.issubdtype(values.dtype, np.integer) else 1.0
        if np.min(values[:, :, 3]) < opaque:
            values = values[:, :, 3]
    return binarizeMask(values, threshold)


def alphaMask(view, params):
    """Mask from the alpha channel of a view, None if the alpha channel carries no mask."""
    _, alpha, _ = readImage(view["path"])
    if alpha is None:
        return None
    mask = alpha > params.maskThreshold
    if mask.all():
        return None  # fully opaque: no mask information
    if params.maskRemoveBorderComponents:
        mask = removeBorderComponents(mask)
    if not mask.any():
        return None
    return mask


def computePoseMask(poseId, views, params, logger):
    """Pose mask at the input image resolution, from (by priority):
    <maskFolder>/<poseId>.png, <maskFolder>/<viewId>.png (vote), <maskFolder>/mask.png (if enabled),
    the alpha channels of the pose images (vote).

    Returns:
        (bool mask or None, description of the source)
    """
    folder = params.maskFolder
    if folder:
        posePath = os.path.join(folder, "{}.png".format(poseId))
        if os.path.isfile(posePath):
            return readMaskFile(posePath, params.maskThreshold), "mask file {}".format(posePath)
        viewPaths = [os.path.join(folder, "{}.png".format(v["viewId"])) for v in views]
        viewPaths = [p for p in viewPaths if os.path.isfile(p)]
        if viewPaths:
            masks = [readMaskFile(p, params.maskThreshold) for p in viewPaths]
            return voteMasks(masks, params.maskVoteThreshold), "vote of {} mask files".format(len(masks))
        globalPath = os.path.join(folder, "mask.png")
        if params.maskUseGlobalFile and os.path.isfile(globalPath):
            return readMaskFile(globalPath, params.maskThreshold), "global mask file {}".format(globalPath)
        logger.warning("Pose {}: no mask file in '{}', using the alpha channels.".format(poseId, folder))
    masks = []
    for view in views:
        mask = alphaMask(view, params)
        if mask is None:
            logger.debug("Pose {}: no mask in the alpha channel of {}".format(poseId, view["path"]))
            continue
        masks.append(mask)
    if not masks:
        return None, "none"
    return voteMasks(masks, params.maskVoteThreshold), "vote of {} alpha masks".format(len(masks))


def resizeMask(mask, size):
    """Resize a bool mask to size = (width, height), aligned on pixel centers like the images.

    The mask is resampled as a float image (area average to shrink, as for the images, bilinear to enlarge)
    and thresholded at 0.5. Nearest neighbour is not used: OpenCV's INTER_NEAREST picks the top-left pixel of
    each block and would shift the mask by (d - 1) / 2 input pixels with respect to the downscaled images.
    """
    if mask.shape[1] == size[0] and mask.shape[0] == size[1]:
        return mask
    import cv2
    shrink = size[0] < mask.shape[1]
    resized = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return resized > 0.5


# --------------------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------------------

def toConvention(normal, convention):
    """Convert OpenGL camera-frame normals (x right, y up, z towards the camera) to the output convention."""
    if convention == "opengl":
        return normal
    if convention == "opencv":
        return normal * np.array([1.0, -1.0, -1.0], np.float32)
    raise ValueError("Unknown normal convention: '{}'".format(convention))


def writeMap(path, values, outputFormat, signed):
    """Write a 1 or 3-channel map, as 16-bit PNG or float32 EXR (raw values).

    For a 16-bit PNG, signed values in [-1, 1] are stored as (v + 1) / 2 and unsigned values are clipped to
    [0, 1]; values are rounded to the nearest integer.
    """
    import cv2
    values = np.asarray(values, np.float32)
    if values.ndim == 3:
        values = values[:, :, ::-1]  # RGB -> BGR
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if outputFormat == "exr":
        ok = cv2.imwrite(path, np.ascontiguousarray(values), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
    elif outputFormat == "png16":
        encoded = (values + 1.0) / 2.0 if signed else values
        encoded = np.round(np.clip(encoded, 0.0, 1.0) * 65535.0).astype(np.uint16)
        ok = cv2.imwrite(path, np.ascontiguousarray(encoded))
    else:
        raise ValueError("Unknown output format: '{}'".format(outputFormat))
    if not ok:
        raise RuntimeError("Cannot write '{}'".format(path))


def writeMask(path, mask):
    """Write a mask as an 8-bit PNG (0/255)."""
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, mask.astype(np.uint8) * 255):
        raise RuntimeError("Cannot write '{}'".format(path))


def finalizeNormals(normal, mask):
    """Zero the normals outside the mask and the undefined ones; returns (normals, support mask)."""
    normal = np.nan_to_num(np.asarray(normal, np.float32))
    norm = np.linalg.norm(normal, axis=2)
    support = norm > 0.5
    if mask is not None:
        support &= mask
    normal = np.where(support[:, :, None], normal / np.maximum(norm, 1e-8)[:, :, None], 0.0).astype(np.float32)
    return normal, support


def checkFailurePolicy(policy, nbProcessed, failedPoses):
    if not failedPoses:
        return
    message = "{} pose(s) could not be processed: {}".format(len(failedPoses), ", ".join(failedPoses))
    if policy == "anyPose" or (policy == "noPose" and nbProcessed == 0):
        raise RuntimeError(message + " (failure policy: {}).".format(policy))


# --------------------------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------------------------

def seedEverything(seed):
    """Seed Python, NumPy and (if available) PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def processPoses(chunk, predict, extraMaps=(), cleanup=None, withNormals=True):
    """Run a photometric stereo method on every multi-lighting pose of the node input.

    Args:
        chunk: the Meshroom node chunk (single chunk).
        predict: function(images, mask) -> {"normal": H x W x 3 OpenGL normals, <extra map>: H x W x C}, with
            images a list of float32 RGB H x W x 3 arrays and mask a bool H x W array (all True without mask).
        extraMaps: names of the other maps returned by predict (written to <outputFolder>/<map>/).
        cleanup: optional function called after each pose (e.g. to free the GPU memory).
        withNormals: False when predict returns no normals: the support of the maps is then the pose mask.
    """
    logger = chunk.logger
    params = Params(chunk.node)
    logger.info("Photometric stereo common layer v{}: {}".format(PS_COMMON_VERSION, params.describe()))
    outputFolder = params.outputFolder
    os.makedirs(outputFolder, exist_ok=True)

    sfm = loadSfm(params.inputSfm, outputFolder)
    sfm, poses = filterPoses(sfm, params.minViewsPerPose, logger)
    # multi-lighting views processed by the node (for inspection, without the landmarks)
    saveSfm({k: v for k, v in sfm.items() if k != "structure"}, os.path.join(outputFolder, "photometricStereoViews.sfm"))
    logger.info("{} multi-lighting pose(s) to process.".format(len(poses)))

    extension = ".exr" if params.outputFormat == "exr" else ".png"
    mapPaths = {name: {} for name in ("normal",) + tuple(extraMaps)}
    processed = []
    failed = []
    for index, (poseId, views) in enumerate(poses.items()):
        start = time.time()
        try:
            if representativeView(poseId, views) is None:
                raise RuntimeError("no view with viewId == poseId to hold the pose in the output SfMData")
            seed = poseSeed(params.seed, poseId)
            selected = selectViews(views, params.nbImages, params.imageSelection, seed)
            images = loadImages(selected, params)
            height, width = images[0].shape[:2]
            mask, source = computePoseMask(poseId, views, params, logger)
            if mask is not None:
                mask = resizeMask(mask, (width, height))
                if not mask.any():
                    raise RuntimeError("empty pose mask ({})".format(source))
            logger.info("Pose {} ({}/{}): {} of {} images, {}x{}, mask: {}".format(
                poseId, index + 1, len(poses), len(selected), len(views), width, height, source))

            seedEverything(seed)
            maps = predict(images, mask if mask is not None else np.ones((height, width), bool))
            if withNormals:
                normal = maps["normal"]
                if normal.shape != (height, width, 3):
                    raise RuntimeError("the method returned normals of shape {} for images of shape {}".format(
                        normal.shape, (height, width, 3)))
                normal, support = finalizeNormals(normal, mask)
                normalPath = os.path.join(outputFolder, poseId + extension)
                writeMap(normalPath, toConvention(normal, params.normalConvention), params.outputFormat, signed=True)
                mapPaths["normal"][poseId] = normalPath
            else:
                support = mask if mask is not None else np.ones((height, width), bool)
            writeMask(os.path.join(params.outputMaskFolder, poseId + ".png"), support)
            for name in extraMaps:
                values = np.asarray(maps[name], np.float32)
                values = values * support.reshape(support.shape + (1,) * (values.ndim - 2))
                path = os.path.join(outputFolder, name, poseId + extension)
                writeMap(path, values, params.outputFormat, signed=False)
                mapPaths[name][poseId] = path
            processed.append(poseId)
            logger.info("Pose {}: done in {:.1f}s ({:.1f}% of the image with normals)".format(
                poseId, time.time() - start, 100.0 * support.mean()))
        except Exception as exc:
            logger.error("Pose {}: failed: {}\n{}".format(poseId, exc, traceback.format_exc()))
            failed.append(poseId)
        finally:
            if cleanup:
                cleanup()

    sfmOutputs = {"normal": chunk.node.outputSfmDataNormal.value} if withNormals else {}
    for name in extraMaps:
        sfmOutputs[name] = getattr(chunk.node, "outputSfmData" + name.capitalize()).value
    for name, sfmPath in sfmOutputs.items():
        saveSfm(buildOutputSfm(sfm, mapPaths[name], params.downscale, params.keepLandmarks), sfmPath)
        logger.info("Saved {} ({} pose(s))".format(sfmPath, len(mapPaths[name])))

    logger.info("{} of {} pose(s) processed.".format(len(processed), len(poses)))
    checkFailurePolicy(params.failurePolicy, len(processed), failed)
