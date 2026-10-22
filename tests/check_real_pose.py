"""Minimal GPU check of the node on one real pose (outside Meshroom): runs the common layer + the method on one
pose of an SfMData and checks the outputs.

Usage (plugin venv, Meshroom on the PYTHONPATH, run from the plugin root):
    python tests/check_real_pose.py <sfm> <poseId> <outputFolder> [--downscale 2] [--nbImages 10] [--set name=value]

Method options (--set): cropMargin, maxProcessingSize, outputInterpolation, normalModel (normal|pbr) and
estimateAlbedo (0|1, albedo of the PBR network, weights/lino_pbr.pth).

Checks:
- output normal map and mask have the downscaled image size, normals are unit or zero,
- the normals cover the pose mask (alignment of the prediction with the image),
- frame: on the silhouette, the normals point outwards, with y up (OpenGL), z towards the camera,
- with estimateAlbedo: the albedo map has the normal map size, is zero outside the mask and in [0, 1] inside,
  and its SfMData has the same views as the normal maps SfMData.
"""
import argparse
import glob
import importlib.util
import json
import logging
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_psCommon import FakeChunk, FakeNode, ps  # noqa: E402
from test_LINOUniPS import lino  # noqa: E402


def makePredict(node, extra):
    """Method-specific part: LINO-UniPS (normal and/or PBR network, as the node)."""
    import torch
    import meshroom_predict as api
    normalModel = extra.get("normalModel", "normal")
    estimateAlbedo = extra.get("estimateAlbedo", "0").lower() in ("1", "true")
    predictors = {}
    for model in lino.requiredModels(normalModel, estimateAlbedo):
        weights = os.path.join(ROOT, "weights", lino.WEIGHTS_FILES[model])
        predictors[model] = api.loadModel(weights, useGpu=True, logger=logging.getLogger(), model=model)
    predict = lino.makePredict(api, predictors, normalModel, estimateAlbedo,
                               cropMargin=int(extra.get("cropMargin", 8)),
                               maxProcessingSize=int(extra.get("maxProcessingSize", 6000)),
                               outputInterpolation=extra.get("outputInterpolation", "cubic"))
    return predict, lino.EXTRA_MAPS if estimateAlbedo else (), torch.cuda.empty_cache


def outwardAgreement(normal, mask):
    """Mean cosine between the image-plane normal direction (OpenGL: (nx, -ny) in image coordinates) and the
    outward direction of the silhouette, on the mask boundary."""
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3)
    gy, gx = np.gradient(blurred)
    outward = -np.stack([gx, gy], axis=2)
    boundary = mask & (cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
    n2d = np.stack([normal[:, :, 0], -normal[:, :, 1]], axis=2)
    valid = boundary & (np.linalg.norm(n2d, axis=2) > 0.2) & (np.linalg.norm(outward, axis=2) > 1e-3)
    a = n2d[valid] / np.linalg.norm(n2d[valid], axis=1, keepdims=True)
    b = outward[valid] / np.linalg.norm(outward[valid], axis=1, keepdims=True)
    return float(np.mean(np.sum(a * b, axis=1))), int(valid.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sfm")
    parser.add_argument("poseId")
    parser.add_argument("output")
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--nbImages", type=int, default=10)
    parser.add_argument("--set", action="append", default=[], help="method option name=value")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extra = dict(s.split("=", 1) for s in args.set)

    os.makedirs(args.output, exist_ok=True)
    sfm = ps.loadSfm(args.sfm, args.output)
    sfm["views"] = [v for v in sfm["views"] if str(v["poseId"]) == args.poseId]
    sfm["poses"] = [p for p in sfm["poses"] if str(p["poseId"]) == args.poseId]
    subset = os.path.join(args.output, "pose.sfm")
    ps.saveSfm(sfm, subset)

    node = FakeNode(args.output, inputSfm=subset, downscale=args.downscale, nbImages=args.nbImages)
    predict, extraMaps, cleanup = makePredict(node, extra)
    for name in extraMaps:
        setattr(node, "outputSfmData" + name.capitalize(),
                type(node.outputFolder)(os.path.join(args.output, name + "Maps.sfm")))
    ps.processPoses(FakeChunk(node), predict, extraMaps=extraMaps, cleanup=cleanup)

    normal = cv2.imread(os.path.join(args.output, args.poseId + ".png"), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    normal = normal.astype(np.float32) / 65535 * 2 - 1
    support = cv2.imread(os.path.join(args.output, "masks", args.poseId + ".png"), cv2.IMREAD_UNCHANGED) > 0
    views, _ = ps.filterPoses(sfm, 1, logging.getLogger())[1][args.poseId], None
    image = ps.downscaleImage(ps.readImage(views[0]["path"])[0], args.downscale)
    poseMask, source = ps.computePoseMask(args.poseId, views, ps.Params(node), logging.getLogger())
    poseMask = ps.resizeMask(poseMask, (image.shape[1], image.shape[0]))
    norms = np.linalg.norm(normal, axis=2)
    cosine, nbBoundary = outwardAgreement(normal, poseMask)
    report = {
        "imageShape": list(image.shape[:2]), "normalShape": list(normal.shape[:2]),
        "maskSource": source, "maskPixels": int(poseMask.sum()),
        "maskCoverage": float(support[poseMask].mean()), "normalsOutsideMask": int((support & ~poseMask).sum()),
        "normRange": [float(norms[support].min()), float(norms[support].max())],
        "meanNormal": normal[support].mean(axis=0).round(3).tolist(),
        "outwardCosine": round(cosine, 3), "boundaryPixels": nbBoundary,
    }
    if "albedo" in extraMaps:
        albedo = cv2.imread(os.path.join(args.output, "albedo", args.poseId + ".png"), cv2.IMREAD_UNCHANGED)
        albedo = albedo[:, :, ::-1].astype(np.float32) / 65535
        albedoSfm = json.load(open(os.path.join(args.output, "albedoMaps.sfm")))
        normalSfm = json.load(open(os.path.join(args.output, "normalMaps.sfm")))
        report.update({
            "albedoShape": list(albedo.shape[:2]), "albedoOutsideMask": int(albedo[~support].any(axis=1).sum()),
            "albedoRange": [float(albedo[support].min()), float(albedo[support].max())],
            "meanAlbedo": albedo[support].mean(axis=0).round(3).tolist(),
            "albedoSfmViews": sorted(v["viewId"] for v in albedoSfm["views"]),
            "normalSfmViews": sorted(v["viewId"] for v in normalSfm["views"]),
        })
    print(json.dumps(report, indent=1))
    if "albedo" in extraMaps:
        assert report["albedoShape"] == report["normalShape"]
        assert report["albedoOutsideMask"] == 0
        assert report["albedoSfmViews"] == report["normalSfmViews"] == [args.poseId]
        assert report["albedoRange"][1] > report["albedoRange"][0], "constant albedo"
    assert report["imageShape"] == report["normalShape"]
    assert report["maskCoverage"] > 0.98, "normals do not cover the mask"
    assert report["normalsOutsideMask"] == 0
    assert abs(report["normRange"][0] - 1) < 1e-3 and abs(report["normRange"][1] - 1) < 1e-3
    # a frame error (flipped x or y) gives a low or negative cosine; real objects stay below 1 (concavities,
    # occlusion boundaries): LINO 0.56 and Uni-MS-PS 0.49 on the reference warrior pose
    assert report["outwardCosine"] > 0.4, "silhouette normals do not point outwards in the OpenGL frame"
    assert report["meanNormal"][2] > 0.3, "normals do not point towards the camera"
    print("CHECK OK")


if __name__ == "__main__":
    main()
