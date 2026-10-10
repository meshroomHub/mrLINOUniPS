"""LINOUniPS node: albedo output (LINO-UniPS PBR network) and choice of the normal network, CPU only.

The LINO_UniPS predict API (meshroom_predict) is replaced by FakeApi: no network, no weights.
Run from the plugin root, with Meshroom on the PYTHONPATH:
    PYTHONPATH=<Meshroom> venv/bin/python -m pytest tests
"""
import importlib
import importlib.util
import json
import logging
import os
import sys

import cv2
import numpy as np
import pytest

from test_psCommon import _scene

PACKAGE = "linoUniPsNodes"
FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meshroom", "LINOUniPS")
LOGGER = logging.getLogger("test_LINOUniPS")
ALBEDO = (0.2, 0.4, 0.6)


def loadNodeModule():
    """Import meshroom/LINOUniPS/LINOUniPS.py as a submodule of a package (the node uses relative imports)."""
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(PACKAGE, os.path.join(FOLDER, "__init__.py"),
                                                      submodule_search_locations=[FOLDER])
        package = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = package
        spec.loader.exec_module(package)
    return importlib.import_module(PACKAGE + ".LINOUniPS")


lino = loadNodeModule()


class Value:
    def __init__(self, value):
        self.value = value
        self.evalValue = value


class FakeNode:
    """Node with the attributes of the LINOUniPS description at their default values ("{nodeCacheFolder}" output
    expressions resolved in the cache folder)."""

    def __init__(self, cache, **values):
        for attribute in lino.LINOUniPS.inputs:
            setattr(self, attribute.name, Value(attribute.value))
        self.linoUniPsPath.evalValue = ""
        for attribute in lino.LINOUniPS.outputs:
            value = attribute.value
            if isinstance(value, str):
                value = value.replace("{nodeCacheFolder}", str(cache))
            setattr(self, attribute.name, Value(value))
        for name, value in values.items():
            getattr(self, name).value = value


class FakeLogManager:
    def start(self, level):
        pass

    def end(self):
        pass


class FakeChunk:
    def __init__(self, node):
        self.node = node
        self.logger = LOGGER
        self.logManager = FakeLogManager()


class FakeApi:
    """Stand-in for meshroom_predict: the "normal" network predicts +z normals, the "pbr" network tilted normals
    and a constant albedo."""

    def __init__(self):
        self.loaded = []
        self.calls = []

    def loadModel(self, weightsPath, useGpu=True, logger=None, model="normal"):
        self.loaded.append((model, weightsPath))
        return model

    def predict(self, predictor, images, mask, **options):
        self.calls.append((predictor, options))
        normal = np.zeros(images[0].shape, np.float32)
        normal[mask] = [0.0, 0.0, 1.0] if predictor == "normal" else [0.0, 0.6, 0.8]
        maps = {"normal": normal}
        if predictor == "pbr":
            albedo = np.zeros(images[0].shape, np.float32)
            albedo[mask] = ALBEDO
            maps.update({"albedo": albedo, "roughness": np.full(mask.shape, 0.5, np.float32),
                         "metallic": np.zeros(mask.shape, np.float32)})
        return maps


# ---------------------------------------------------------------------------------------------- description

def attributes(name):
    return {a.name: a for a in getattr(lino.LINOUniPS, name)}


def test_albedo_attributes_defaults_keep_normal_only_behaviour():
    inputs = attributes("inputs")
    assert inputs["estimateAlbedo"].value is False and not inputs["estimateAlbedo"].advanced
    assert inputs["normalModel"].value == "normal" and inputs["normalModel"].values == ["normal", "pbr"]
    assert inputs["pbrModelPath"].value == "" and inputs["pbrModelPath"].advanced
    outputs = attributes("outputs")
    assert outputs["outputSfmDataAlbedo"].value == "{nodeCacheFolder}/albedoMaps.sfm"
    assert outputs["albedoMaps"].semantic == "image"


def test_new_inputs_at_default_keep_the_node_uid(tmp_path):
    """Existing nodes keep their UID (and computed caches): at their default values, the new inputs are ignored
    by the UID (Meshroom skips an attribute whose value is its uidIgnoreValue, or which is disabled)."""
    inputs = attributes("inputs")
    assert inputs["estimateAlbedo"].checkUidIgnoreValue(False) and not inputs["estimateAlbedo"].checkUidIgnoreValue(True)
    assert inputs["normalModel"].checkUidIgnoreValue("normal") and not inputs["normalModel"].checkUidIgnoreValue("pbr")
    assert not inputs["pbrModelPath"].enabled(FakeNode(tmp_path))


@pytest.mark.parametrize("normalModel,estimateAlbedo,models", [
    ("normal", False, ["normal"]),
    ("pbr", False, ["pbr"]),
    ("pbr", True, ["pbr"]),
    ("normal", True, ["normal", "pbr"]),
])
def test_required_models_and_enabled_attributes(tmp_path, normalModel, estimateAlbedo, models):
    assert lino.requiredModels(normalModel, estimateAlbedo) == models
    node = FakeNode(tmp_path, normalModel=normalModel, estimateAlbedo=estimateAlbedo)
    inputs, outputs = attributes("inputs"), attributes("outputs")
    assert inputs["modelPath"].enabled(node) == ("normal" in models)
    assert inputs["pbrModelPath"].enabled(node) == ("pbr" in models)
    assert outputs["outputSfmDataAlbedo"].enabled(node) == estimateAlbedo
    assert outputs["albedoMaps"].enabled(node) == estimateAlbedo


# ---------------------------------------------------------------------------------------------- predict

def poseInputs():
    images = [np.full((6, 8, 3), 0.5, np.float32)] * 3
    mask = np.zeros((6, 8), bool)
    mask[1:5, 2:6] = True
    return images, mask


def test_predict_pbr_model_gives_normals_and_albedo_in_one_pass():
    api = FakeApi()
    predict = lino.makePredict(api, {"pbr": "pbr"}, "pbr", True, cropMargin=4)
    maps = predict(*poseInputs())
    assert set(maps) == {"normal", "albedo"}
    assert api.calls == [("pbr", {"cropMargin": 4})]
    np.testing.assert_allclose(maps["normal"][2, 3], [0.0, 0.6, 0.8])
    np.testing.assert_allclose(maps["albedo"][2, 3], ALBEDO)


def test_predict_normal_model_with_albedo_runs_both_networks():
    api = FakeApi()
    predict = lino.makePredict(api, {"normal": "normal", "pbr": "pbr"}, "normal", True)
    maps = predict(*poseInputs())
    assert [predictor for predictor, _ in api.calls] == ["normal", "pbr"]
    np.testing.assert_allclose(maps["normal"][2, 3], [0.0, 0.0, 1.0])  # normals of the normal model
    np.testing.assert_allclose(maps["albedo"][2, 3], ALBEDO)


@pytest.mark.parametrize("normalModel", ["normal", "pbr"])
def test_predict_without_albedo_returns_normals_only(normalModel):
    api = FakeApi()
    maps = lino.makePredict(api, {normalModel: normalModel}, normalModel, False)(*poseInputs())
    assert set(maps) == {"normal"}
    assert [predictor for predictor, _ in api.calls] == [normalModel]


# ---------------------------------------------------------------------------------------------- weights

@pytest.fixture
def isolatedWeights(tmp_path, monkeypatch):
    """Default weights locations in tmp_path: plugin weights/, LINO_UniPS weights/ and the torch hub cache."""
    nodeFolder = tmp_path / "plugin" / "meshroom" / "LINOUniPS"
    nodeFolder.mkdir(parents=True)  # the plugin weights are searched in <node folder>/../../weights
    monkeypatch.setattr(lino, "__file__", str(nodeFolder / "LINOUniPS.py"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    node = FakeNode(tmp_path / "cache")
    node.linoUniPsPath.evalValue = str(tmp_path / "LINO_UniPS")
    return node, tmp_path


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


def test_find_pbr_weights_default_locations(isolatedWeights):
    node, root = isolatedWeights
    with pytest.raises(RuntimeError, match="download_weights.sh pbr"):
        lino.LINOUniPS.findWeights(node, "pbr")
    hub = touch(root / "home" / ".cache" / "torch" / "hub" / "checkpoints" / "lino_pbr.pth")
    assert lino.LINOUniPS.findWeights(node, "pbr") == hub
    linoRepo = touch(root / "LINO_UniPS" / "weights" / "lino_pbr.pth")
    assert lino.LINOUniPS.findWeights(node, "pbr") == linoRepo
    plugin = touch(root / "plugin" / "weights" / "lino_pbr.pth")
    assert lino.LINOUniPS.findWeights(node, "pbr") == plugin
    # the normal model keeps its own file
    with pytest.raises(RuntimeError, match="normal weights not found"):
        lino.LINOUniPS.findWeights(node, "normal")


def test_find_weights_from_the_model_inputs(isolatedWeights):
    node, _ = isolatedWeights
    node.modelPath.value = "/weights/custom.pth"
    node.pbrModelPath.value = "/weights/custom_pbr.pth"
    assert lino.LINOUniPS.findWeights(node, "normal") == "/weights/custom.pth"
    assert lino.LINOUniPS.findWeights(node, "pbr") == "/weights/custom_pbr.pth"


# ---------------------------------------------------------------------------------------------- node outputs

def runNode(tmp_path, monkeypatch, **values):
    cache = tmp_path / "cache"
    node = FakeNode(cache, inputSfm=_scene(tmp_path), downscale=2, **values)
    api = FakeApi()
    monkeypatch.setattr(lino.LINOUniPS, "importApi", staticmethod(lambda node: api))
    monkeypatch.setattr(lino.LINOUniPS, "findWeights", staticmethod(lambda node, model: "/weights/" + model))
    lino.LINOUniPS().processChunk(FakeChunk(node))
    return cache, api


def test_node_default_outputs_normals_only(tmp_path, monkeypatch):
    cache, api = runNode(tmp_path, monkeypatch)
    assert api.loaded == [("normal", "/weights/normal")]
    assert (cache / "normalMaps.sfm").exists() and (cache / "100.png").exists()
    assert not (cache / "albedoMaps.sfm").exists() and not (cache / "albedo").exists()


@pytest.mark.parametrize("normalModel,loaded", [("pbr", ["pbr"]), ("normal", ["normal", "pbr"])])
def test_node_albedo_outputs_for_rnb_neus2(tmp_path, monkeypatch, normalModel, loaded):
    cache, api = runNode(tmp_path, monkeypatch, estimateAlbedo=True, normalModel=normalModel)
    assert [model for model, _ in api.loaded] == loaded
    normals = json.loads((cache / "normalMaps.sfm").read_text())
    albedos = json.loads((cache / "albedoMaps.sfm").read_text())
    # same views, poses and (downscaled) intrinsics as the normal maps: one albedo map per pose
    assert sorted(v["viewId"] for v in albedos["views"]) == ["100", "200"]
    assert [p["poseId"] for p in albedos["poses"]] == [p["poseId"] for p in normals["poses"]]
    assert albedos["intrinsics"] == normals["intrinsics"]
    for view in albedos["views"]:
        assert view["path"] == str(cache / "albedo" / (view["viewId"] + ".png"))
    normalViews = {v["viewId"]: dict(v, path=None) for v in normals["views"]}
    assert all(dict(v, path=None) == normalViews[v["viewId"]] for v in albedos["views"])
    # 16-bit RGB albedo at the normal map resolution, zero outside the pose mask
    albedo = cv2.imread(str(cache / "albedo" / "100.png"), cv2.IMREAD_UNCHANGED)
    normal = cv2.imread(str(cache / "100.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(cache / "masks" / "100.png"), cv2.IMREAD_UNCHANGED) > 0
    assert albedo.dtype == np.uint16 and albedo.shape == normal.shape == (10, 15, 3)
    assert mask.any() and np.all(albedo[mask] == np.round(np.array(ALBEDO[::-1]) * 65535))  # BGR on disk
    assert not albedo[~mask].any()


def test_node_albedo_exr_output(tmp_path, monkeypatch):
    cache, _ = runNode(tmp_path, monkeypatch, estimateAlbedo=True, normalModel="pbr", outputFormat="exr")
    albedo = cv2.imread(str(cache / "albedo" / "100.exr"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(cache / "masks" / "100.png"), cv2.IMREAD_UNCHANGED) > 0
    assert albedo.dtype == np.float32 and albedo.shape == (10, 15, 3)
    np.testing.assert_allclose(albedo[mask], np.tile(ALBEDO[::-1], (mask.sum(), 1)), atol=1e-6)
