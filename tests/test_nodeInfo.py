"""Node metadata shown in the Meshroom node info: plugin author and license, credits of the wrapped method."""
import ast
import os

NODE_FILE = os.path.join(os.path.dirname(__file__), "..", "meshroom/LINOUniPS/LINOUniPS.py")


def _parse():
    with open(NODE_FILE) as f:
        return ast.parse(f.read())


def _moduleValue(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("{} is not defined at module level".format(name))


def _nodeInfo(tree):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LINOUniPS":
            for item in node.body:
                if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__nodeInfo__" for t in item.targets):
                    return dict(ast.literal_eval(item.value))
    raise AssertionError("LINOUniPS.__nodeInfo__ is not defined")


def test_plugin_author_and_license():
    tree = _parse()
    assert _moduleValue(tree, "__author__") == "Baptiste Brument"
    assert _moduleValue(tree, "__license__") == "MPL-2.0"
    assert _moduleValue(tree, "__version__")


def test_method_credits():
    info = _nodeInfo(_parse())
    assert set(info) >= {"method", "methodLicense", "methodRepository"}
    assert "MIT" in info["methodLicense"]
    assert info["methodRepository"] == "https://github.com/houyuanchen111/LINO_UniPS"
