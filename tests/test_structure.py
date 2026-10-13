"""Offline structural checks; runnable with an empty Python 3.14 environment."""

import ast
import json
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StructureTests(unittest.TestCase):
    def test_package_syntax(self):
        for path in (ROOT / "pokeml").rglob("*.py"):
            with self.subTest(path=path):
                ast.parse(path.read_text(), filename=str(path))

    def test_notebook_code_syntax_and_kernel(self):
        for path in (ROOT / "notebooks").rglob("*.ipynb"):
            notebook = json.loads(path.read_text())
            self.assertEqual(notebook["metadata"]["language_info"]["version"], "3.14.8")
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    with self.subTest(path=path, cell=cell.get("id")):
                        ast.parse("".join(cell["source"]))
                        self.assertFalse(any(output.get('output_type') == 'error'
                                             for output in cell['outputs']))
            if notebook['metadata'].get('pokeml_dataset_sha256'):
                import hashlib
                self.assertEqual(notebook['metadata']['pokeml_dataset_sha256'],
                                 hashlib.sha256((ROOT / 'datasets/pkdx_min.csv').read_bytes()).hexdigest())

    def test_local_import_modules_exist(self):
        sources = [p.read_text() for p in (ROOT / "pokeml").rglob("*.py")]
        sources += ["".join(c["source"]) for p in (ROOT / "notebooks").rglob("*.ipynb")
                    for c in json.loads(p.read_text())["cells"] if c["cell_type"] == "code"]
        for source in sources:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pokeml"):
                    module = ROOT.joinpath(*node.module.split("."))
                    with self.subTest(module=node.module):
                        self.assertTrue(module.with_suffix(".py").exists() or
                                        (module / "__init__.py").exists())

    def test_requirements_match_package(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        requirements = [line for line in (ROOT / "requirements.txt").read_text().splitlines()
                        if line and not line.startswith("#")]
        self.assertEqual(requirements, ["-e .[notebooks,dev]"])
        extras = project["optional-dependencies"]
        self.assertEqual(set(extras), {"notebooks", "dev"})
        self.assertTrue(any(item.startswith("nbclient") for item in extras["notebooks"]))
        self.assertTrue(any(item.startswith("pytest") for item in extras["dev"]))
        self.assertTrue((ROOT / "requirements/lock-py314-macos.txt").is_file())
        self.assertEqual(project["requires-python"], ">=3.14,<3.15")

    def test_workflow_has_no_acquire_or_clean_import(self):
        source = ast.parse((ROOT / "pokeml/pipeline/workflow.py").read_text())
        imports = {n.module for n in ast.walk(source) if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("pokeml.data.acquire", imports)
        self.assertNotIn("pokeml.data.clean", imports)


if __name__ == "__main__":
    unittest.main()
