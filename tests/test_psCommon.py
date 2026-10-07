"""Unit tests of the common photometric stereo layer (psCommon.py), CPU only.

Run from the plugin root, with Meshroom on the PYTHONPATH:
    PYTHONPATH=<Meshroom> venv/bin/python -m pytest tests
"""
import glob
import importlib.util
import json
import logging
import os

import cv2
import numpy as np
import pytest

_PATH = glob.glob(os.path.join(os.path.dirname(__file__), "..", "meshroom", "*", "psCommon.py"))[0]
_SPEC = importlib.util.spec_from_file_location("psCommon", _PATH)
ps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ps)

LOGGER = logging.getLogger("test_psCommon")


class _Value:
    def __init__(self, value):
        self.value = value


class FakeNode:
    """Node with the common attributes set to their default values."""

    def __init__(self, cache, **values):
        attributes = ps.inputAttributes() + ps.advancedInputAttributes() + ps.settingsAttributes()
        for attribute in attributes:
            setattr(self, attribute.name, _Value(attribute.value))
        self.outputFolder = _Value(str(cache))
        self.outputMaskFolder = _Value(os.path.join(str(cache), "masks"))
        self.outputSfmDataNormal = _Value(os.path.join(str(cache), "normalMaps.sfm"))
        for name, value in values.items():
            getattr(self, name).value = value


class FakeChunk:
    def __init__(self, node):
        self.node = node
        self.logger = LOGGER


def params(tmp_path, **values):
    return ps.Params(FakeNode(tmp_path, **values))


# ---------------------------------------------------------------------------------------------- masks

def test_vote_strict_majority():
    full = np.ones((4, 4), bool)
    partial = full.copy()
    partial[:2] = False
    combined = ps.voteMasks([full, full, partial], 0.5)
    assert combined.all()  # 2 of 3 masks: kept
    assert not ps.voteMasks([full, partial], 0.5)[:2].any()  # 1 of 2: not a strict majority


def test_vote_intersection_and_union():
    a = np.zeros((2, 2), bool)
    a[0, 0] = True
    b = np.zeros((2, 2), bool)
    b[0, 0] = b[1, 1] = True
    assert ps.voteMasks([a, b], 1.0).sum() == 1
    assert ps.voteMasks([a, b], 0.0).sum() == 2


def test_binarize_value_ranges():
    assert ps.binarizeMask(np.array([[100, 200]], np.uint8), 0.5).tolist() == [[False, True]]
    assert ps.binarizeMask(np.array([[30000, 40000]], np.uint16), 0.5).tolist() == [[False, True]]
    assert ps.binarizeMask(np.array([[0.4, 0.6]], np.float32), 0.5).tolist() == [[False, True]]
    assert ps.binarizeMask(np.full((1, 1, 3), 255, np.uint8), 0.5).tolist() == [[True]]


def test_remove_border_components():
    mask = np.zeros((10, 10), bool)
    mask[0:3, 0:3] = True  # touches the border
    mask[5:8, 5:8] = True  # interior object
    cleaned = ps.removeBorderComponents(mask)
    assert not cleaned[0:3, 0:3].any() and cleaned[5:8, 5:8].all()


def _writeRgba(path, alpha):
    rgba = np.zeros(alpha.shape + (4,), np.uint8)
    rgba[:, :, :3] = 128
    rgba[:, :, 3] = alpha
    cv2.imwrite(str(path), rgba)
    return {"viewId": os.path.splitext(os.path.basename(str(path)))[0], "path": str(path)}


def _objectAlpha(shape=(20, 30), top=5):
    alpha = np.zeros(shape, np.uint8)
    alpha[top:15, 8:22] = 255
    return alpha


def test_pose_mask_from_alpha_vote(tmp_path):
    views = [_writeRgba(tmp_path / "{}.png".format(i), _objectAlpha()) for i in range(4)]
    views.append(_writeRgba(tmp_path / "4.png", _objectAlpha(top=11)))  # partial mask: outvoted
    views.append(_writeRgba(tmp_path / "5.png", np.full((20, 30), 255, np.uint8)))  # opaque: ignored
    mask, source = ps.computePoseMask("7", views, params(tmp_path), LOGGER)
    assert source == "vote of 5 alpha masks"
    assert mask.sum() == 10 * 14 and mask[5:15, 8:22].all()
    andMask, _ = ps.computePoseMask("7", views, params(tmp_path, maskVoteThreshold=1.0), LOGGER)
    assert andMask.sum() == 4 * 14


def test_pose_mask_border_components_option(tmp_path):
    alpha = _objectAlpha()
    alpha[:, :3] = 255  # undistortion-like band on the border
    views = [_writeRgba(tmp_path / "{}.png".format(i), alpha) for i in range(3)]
    mask, _ = ps.computePoseMask("7", views, params(tmp_path), LOGGER)
    assert not mask[:, :3].any()
    mask, _ = ps.computePoseMask("7", views, params(tmp_path, maskRemoveBorderComponents=False), LOGGER)
    assert mask[:, :3].all()


def test_pose_mask_file_priority(tmp_path):
    folder = tmp_path / "masks"
    folder.mkdir()
    views = [_writeRgba(tmp_path / "{}.png".format(i), _objectAlpha()) for i in range(3)]
    p = params(tmp_path, maskFolder=str(folder))
    # no file: alpha
    assert ps.computePoseMask("7", views, p, LOGGER)[1].startswith("vote of 3 alpha")
    # global file: only when enabled
    cv2.imwrite(str(folder / "mask.png"), np.full((20, 30), 255, np.uint8))
    assert ps.computePoseMask("7", views, p, LOGGER)[1].startswith("vote of 3 alpha")
    assert ps.computePoseMask("7", views, params(tmp_path, maskFolder=str(folder), maskUseGlobalFile=True),
                              LOGGER)[1].startswith("global")
    # per view files
    for v in views[:2]:
        cv2.imwrite(str(folder / "{}.png".format(v["viewId"])), _objectAlpha())
    mask, source = ps.computePoseMask("7", views, p, LOGGER)
    assert source == "vote of 2 mask files" and mask.sum() == 10 * 14
    # pose file
    cv2.imwrite(str(folder / "7.png"), np.full((20, 30), 255, np.uint8))
    mask, source = ps.computePoseMask("7", views, p, LOGGER)
    assert source.startswith("mask file") and mask.all()


def test_no_mask(tmp_path):
    views = [_writeRgba(tmp_path / "{}.png".format(i), np.full((20, 30), 255, np.uint8)) for i in range(3)]
    assert ps.computePoseMask("7", views, params(tmp_path), LOGGER) == (None, "none")


@pytest.mark.parametrize("d", [2, 3, 4])
def test_resize_mask_aligned_with_images(d):
    """A downscaled mask must coincide with the same object in the downscaled images (no half-pixel shift)."""
    rng = np.random.default_rng(d)
    mask = np.zeros((12 * d, 16 * d), bool)
    for _ in range(20):
        r, c = rng.integers(0, 10) * d, rng.integers(0, 14) * d
        mask[r:r + 2 * d, c:c + 2 * d] = True  # blocks aligned on the downscale grid
    image = ps.downscaleImage(np.repeat(mask[:, :, None].astype(np.float32), 3, axis=2), d)
    small = ps.resizeMask(mask, (16, 12))
    assert small.shape == (12, 16) and small.dtype == bool
    assert (small == (image[:, :, 0] > 0.5)).all()


def test_resize_mask_centroid_preserved():
    mask = np.zeros((40, 40), bool)
    mask[10:17, 21:30] = True  # odd sizes, not aligned on the grid
    small = ps.resizeMask(mask, (20, 20))
    rows, cols = np.nonzero(mask)
    srows, scols = np.nonzero(small)
    # centers: (x + 0.5) / 2 - 0.5 in the output pixel grid
    assert abs(srows.mean() - ((rows.mean() + 0.5) / 2 - 0.5)) < 0.3
    assert abs(scols.mean() - ((cols.mean() + 0.5) / 2 - 0.5)) < 0.3


# ---------------------------------------------------------------------------------------------- images

def test_select_views_modes():
    views = [{"path": str(i)} for i in range(10)]
    assert ps.selectViews(views, -1, "random", 0) == views
    assert ps.selectViews(views, 20, "random", 0) == views
    assert [v["path"] for v in ps.selectViews(views, 3, "first", 0)] == ["0", "1", "2"]
    assert [v["path"] for v in ps.selectViews(views, 4, "uniform", 0)] == ["0", "3", "6", "9"]
    a = ps.selectViews(views, 4, "random", 123)
    assert a == ps.selectViews(views, 4, "random", 123)  # reproducible
    assert [int(v["path"]) for v in a] == sorted(int(v["path"]) for v in a)  # order kept


def test_pose_seed_depends_only_on_seed_and_pose():
    assert ps.poseSeed(42, "123") == ps.poseSeed(42, "123")
    assert ps.poseSeed(42, "123") != ps.poseSeed(42, "124")
    assert ps.poseSeed(42, "123") != ps.poseSeed(43, "123")


def test_read_image_formats(tmp_path):
    bgr8 = np.zeros((2, 3, 3), np.uint8)
    bgr8[:, :, 2] = 255  # red
    cv2.imwrite(str(tmp_path / "a.png"), bgr8)
    rgb, alpha, isFloat = ps.readImage(str(tmp_path / "a.png"))
    assert rgb.dtype == np.float32 and alpha is None and not isFloat
    assert np.allclose(rgb[0, 0], [1, 0, 0])

    bgra16 = np.zeros((2, 3, 4), np.uint16)
    bgra16[:, :, 0] = 65535  # blue
    bgra16[:, :, 3] = 32768
    cv2.imwrite(str(tmp_path / "b.png"), bgra16)
    rgb, alpha, _ = ps.readImage(str(tmp_path / "b.png"))
    assert np.allclose(rgb[0, 0], [0, 0, 1]) and np.allclose(alpha, 32768 / 65535)

    exr = np.zeros((2, 3, 3), np.float32)
    exr[:, :, 1] = 2.5  # green, > 1 kept
    cv2.imwrite(str(tmp_path / "c.exr"), exr)
    rgb, _, isFloat = ps.readImage(str(tmp_path / "c.exr"))
    assert isFloat and np.allclose(rgb[0, 0], [0, 2.5, 0])

    cv2.imwrite(str(tmp_path / "d.png"), np.full((2, 3), 51, np.uint8))
    rgb, _, _ = ps.readImage(str(tmp_path / "d.png"))
    assert rgb.shape == (2, 3, 3) and np.allclose(rgb, 0.2)


def test_downscale_size():
    image = np.zeros((11, 17, 3), np.float32)
    assert ps.downscaleImage(image, 2).shape == (5, 8, 3)
    assert ps.downscaleImage(image, 1) is image


def test_srgb_to_linear():
    assert np.allclose(ps.srgbToLinear(np.array([0.0, 0.04045, 1.0], np.float32)), [0.0, 0.04045 / 12.92, 1.0])


# ---------------------------------------------------------------------------------------------- outputs

def test_png16_normal_encoding(tmp_path):
    normal = np.array([[[0, 0, 1], [0, 0, 0]]], np.float32)
    path = str(tmp_path / "n.png")
    ps.writeMap(path, normal, "png16", signed=True)
    stored = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]  # RGB
    assert stored.dtype == np.uint16
    assert stored[0, 0].tolist() == [32768, 32768, 65535]
    decoded = stored.astype(np.float32) / 65535.0 * 2.0 - 1.0  # RNb-NeuS2 decoding
    assert np.allclose(decoded, normal, atol=2e-5)


def test_exr_map_roundtrip(tmp_path):
    values = np.random.default_rng(0).uniform(-1, 1, (3, 4, 3)).astype(np.float32)
    path = str(tmp_path / "n.exr")
    ps.writeMap(path, values, "exr", signed=True)
    stored = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    assert np.allclose(stored, values)


def test_single_channel_map(tmp_path):
    path = str(tmp_path / "r.png")
    ps.writeMap(path, np.full((2, 2), 2.0, np.float32), "png16", signed=False)
    assert cv2.imread(path, cv2.IMREAD_UNCHANGED).tolist() == [[65535, 65535], [65535, 65535]]


def test_convention():
    n = np.array([[[0.1, 0.2, 0.97]]], np.float32)
    assert np.allclose(ps.toConvention(n, "opengl"), n)
    assert np.allclose(ps.toConvention(n, "opencv"), [[[0.1, -0.2, -0.97]]])


def test_finalize_normals():
    normal = np.array([[[0, 0, 2], [0, 0, 1], [0.1, 0, 0]]], np.float32)
    mask = np.array([[True, False, True]])
    out, support = ps.finalizeNormals(normal, mask)
    assert support.tolist() == [[True, False, False]]
    assert np.allclose(out[0, 0], [0, 0, 1]) and not out[0, 1:].any()


def test_failure_policy():
    ps.checkFailurePolicy("noPose", 1, ["a"])
    ps.checkFailurePolicy("never", 0, ["a"])
    ps.checkFailurePolicy("anyPose", 3, [])
    with pytest.raises(RuntimeError):
        ps.checkFailurePolicy("noPose", 0, ["a"])
    with pytest.raises(RuntimeError):
        ps.checkFailurePolicy("anyPose", 3, ["a"])


# ---------------------------------------------------------------------------------------------- SfMData

def _sfm():
    views = []
    for pose, nb in (("100", 3), ("200", 4), ("300", 1), ("400", 3)):
        for i in range(nb):
            viewId = pose if i == 0 else str(int(pose) + i)
            views.append({"viewId": viewId, "poseId": pose, "intrinsicId": "1", "path": "/img/{}_{}.png".format(pose, i),
                          "width": "8288", "height": "5520"})
    return {
        "views": views,
        "intrinsics": [{"intrinsicId": "1", "width": "8288", "height": "5520", "principalPoint": ["10.5", "-3"]}],
        "poses": [{"poseId": p} for p in ("100", "200", "300")],  # pose 400 is not localized
        "structure": [{"landmarkId": "0", "X": ["1", "2", "3"], "observations": [{"observationId": "100"}]}],
    }


def test_filter_poses():
    sfm, poses = ps.filterPoses(_sfm(), 3, LOGGER)
    assert list(poses) == ["100", "200"]
    assert {v["poseId"] for v in sfm["views"]} == {"100", "200"}
    assert [p["poseId"] for p in sfm["poses"]] == ["100", "200"]
    with pytest.raises(RuntimeError):
        ps.filterPoses(_sfm(), 5, LOGGER)


def test_output_sfm():
    sfm, _ = ps.filterPoses(_sfm(), 3, LOGGER)
    out = ps.buildOutputSfm(sfm, {"200": "/out/200.png"}, 2, keepLandmarks=True)
    assert [(v["viewId"], v["path"]) for v in out["views"]] == [("200", "/out/200.png")]
    assert [p["poseId"] for p in out["poses"]] == ["200"]
    intrinsic = out["intrinsics"][0]
    assert (intrinsic["width"], intrinsic["height"]) == ("4144", "2760")
    # absolute pp (10.5 + 4144, -3 + 2760) -> (pp + 0.5) / 2 - 0.5, as offsets from the new center (2072, 1380)
    assert [float(v) for v in intrinsic["principalPoint"]] == [5.0, -1.75]
    assert out["views"][0]["width"] == "4144"
    assert out["structure"][0]["X"] == ["1", "2", "3"] and out["structure"][0]["observations"] == []
    assert sfm["structure"][0]["observations"]  # input not modified
    assert "structure" not in ps.buildOutputSfm(sfm, {}, 1, keepLandmarks=False)
    assert sfm["intrinsics"][0]["width"] == "8288"  # input not modified


# ---------------------------------------------------------------------------------------------- main loop

def _scene(tmp_path):
    """Two multi-lighting poses of 3 RGBA images (object mask in alpha) + one single-view pose."""
    views, poses = [], []
    for pose in ("100", "200"):
        poses.append({"poseId": pose})
        for i in range(3):
            viewId = pose if i == 1 else str(int(pose) + 10 + i)
            v = _writeRgba(tmp_path / "{}.png".format(viewId), _objectAlpha())
            v.update({"poseId": pose, "intrinsicId": "1", "width": "30", "height": "20"})
            views.append(v)
    single = _writeRgba(tmp_path / "900.png", _objectAlpha())
    single.update({"poseId": "900", "intrinsicId": "1"})
    views.append(single)
    poses.append({"poseId": "900"})
    sfm = {"views": views, "poses": poses,
           "intrinsics": [{"intrinsicId": "1", "width": "30", "height": "20", "principalPoint": ["0", "0"]}]}
    path = tmp_path / "input.sfm"
    path.write_text(json.dumps(sfm))
    return str(path)


def _flatPredict(images, mask):
    normal = np.zeros(images[0].shape, np.float32)
    normal[mask] = [0.0, 0.6, 0.8]
    return {"normal": normal}


def test_process_poses(tmp_path):
    cache = tmp_path / "cache"
    node = FakeNode(cache, inputSfm=_scene(tmp_path), downscale=2, normalConvention="opencv")
    ps.processPoses(FakeChunk(node), _flatPredict)
    out = json.loads((cache / "normalMaps.sfm").read_text())
    assert sorted(v["viewId"] for v in out["views"]) == ["100", "200"]
    normal = cv2.imread(str(cache / "100.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(cache / "masks" / "100.png"), cv2.IMREAD_UNCHANGED)
    assert normal.shape == (10, 15, 3) and mask.shape == (10, 15)
    assert out["intrinsics"][0]["width"] == "15"
    decoded = normal[:, :, ::-1].astype(np.float32) / 65535 * 2 - 1
    inside = mask > 0
    assert inside.any() and np.allclose(decoded[inside], [0.0, -0.6, -0.8], atol=1e-4)  # opencv convention
    assert np.allclose(decoded[~inside], 0, atol=1e-4)


def test_process_poses_failures(tmp_path):
    cache = tmp_path / "cache"
    sfmPath = _scene(tmp_path)

    def failOnPose100(images, mask):
        failOnPose100.calls += 1
        if failOnPose100.calls == 1:
            raise MemoryError("simulated out of memory")
        return _flatPredict(images, mask)
    failOnPose100.calls = 0

    ps.processPoses(FakeChunk(FakeNode(cache, inputSfm=sfmPath)), failOnPose100)  # noPose: one pose is enough
    assert len(json.loads((cache / "normalMaps.sfm").read_text())["views"]) == 1

    failOnPose100.calls = 0
    with pytest.raises(RuntimeError):
        ps.processPoses(FakeChunk(FakeNode(cache, inputSfm=sfmPath, failurePolicy="anyPose")), failOnPose100)

    def alwaysFail(images, mask):
        raise RuntimeError("simulated failure")
    with pytest.raises(RuntimeError, match="could not be processed"):
        ps.processPoses(FakeChunk(FakeNode(cache, inputSfm=sfmPath)), alwaysFail)


def test_process_poses_without_normals(tmp_path):
    """Methods computing only other maps (e.g. SDM-UniPS BRDF): support = pose mask, no normal output."""
    cache = tmp_path / "cache"
    node = FakeNode(cache, inputSfm=_scene(tmp_path))
    node.outputSfmDataAlbedo = _Value(str(cache / "albedoMaps.sfm"))

    def albedoOnly(images, mask):
        return {"albedo": np.full(images[0].shape, 0.5, np.float32)}
    ps.processPoses(FakeChunk(node), albedoOnly, extraMaps=("albedo",), withNormals=False)
    assert not (cache / "normalMaps.sfm").exists() and not (cache / "100.png").exists()
    out = json.loads((cache / "albedoMaps.sfm").read_text())
    assert sorted(v["viewId"] for v in out["views"]) == ["100", "200"]
    mask = cv2.imread(str(cache / "masks" / "100.png"), cv2.IMREAD_UNCHANGED) > 0
    albedo = cv2.imread(str(cache / "albedo" / "100.png"), cv2.IMREAD_UNCHANGED)
    assert mask.sum() == 10 * 14 and np.all(albedo[mask] == 32768) and not albedo[~mask].any()


def test_output_attributes_enabled():
    outputs = {a.name: a for a in ps.outputAttributes(extraMaps=("albedo",), normalEnabled=False,
                                                      extraEnabled=lambda node: True)}
    assert outputs["normalMaps"].enabled is False and outputs["outputSfmDataNormal"].enabled is False
    assert callable(outputs["albedoMaps"].enabled) and outputs["outputMaskFolder"].enabled is True


def test_process_poses_selection_reproducible(tmp_path):
    seen = []

    def record(images, mask):
        seen.append(len(images))
        return _flatPredict(images, mask)
    node = FakeNode(tmp_path / "cache", inputSfm=_scene(tmp_path), nbImages=2)
    ps.processPoses(FakeChunk(node), record)
    assert seen == [2, 2]


@pytest.mark.parametrize("d,size", [(2, 64), (4, 64), (3, 64), (2, 63)])
def test_downscaled_intrinsics_project_on_downscaled_images(d, size):
    """A point imaged at a known subpixel position (AliceVision: pixel i centered at i) must be found, in the
    image downscaled by downscaleImage, where the downscaled camera model projects it."""
    offset = 3.25
    x = 22.3  # absolute image coordinate of the point in the full image
    cols = np.arange(size, dtype=np.float64)
    image = np.exp(-0.5 * ((cols - x) / 2.0) ** 2)[None, :, None].repeat(4, 0).repeat(3, 2).astype(np.float32)
    small = ps.downscaleImage(image, d)[0, :, 0].astype(np.float64)
    measured = np.sum(np.arange(len(small)) * small) / np.sum(small)
    # camera model: x = f * X / Z + pp, with f scaled by (size // d) / size and pp by scaledPrincipalPoint
    scale = (size // d) / size
    pp = offset + size / 2.0
    ppSmall = ps.scaledPrincipalPoint(offset, size, d) + (size // d) / 2.0
    predicted = (x - pp) * scale + ppSmall
    assert abs(measured - predicted) < 0.02
