"""Check legacy loader calls without loading InsightFace or model files."""

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock


class InsightFaceLoaderTests(unittest.TestCase):
    def test_old_workflows_default_to_buffalo_and_explicit_models_still_work(self):
        path = Path(__file__).resolve().parents[1] / "IPAdapterPlus.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IPAdapterInsightFaceLoader")
        backend = Mock(return_value=object())
        namespace = {"insightface_loader": backend}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
        loader = namespace["IPAdapterInsightFaceLoader"]()
        self.assertNotIn("model_name", loader.INPUT_TYPES()["required"])
        self.assertIn("model_name", loader.INPUT_TYPES()["optional"])
        self.assertEqual(loader.load_insightface("CPU"), (backend.return_value,))
        backend.assert_called_once_with("CPU", model_name="buffalo_l")
        backend.reset_mock()
        loader.load_insightface("CUDA", model_name="antelopev2")
        backend.assert_called_once_with("CUDA", model_name="antelopev2")


if __name__ == "__main__":
    unittest.main()
