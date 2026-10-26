"""Check public node help without importing ComfyUI or loading any models."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


NODE_NAMES = {
    "IPAdapterUnifiedLoader", "IPAdapterUnifiedLoaderFaceID", "IPAdapterUnifiedLoaderCommunity",
    "IPAdapterModelLoader", "IPAdapterInsightFaceLoader", "IPAdapterSimple", "IPAdapterAdvanced",
    "IPAdapterBatch", "IPAdapterStyleComposition", "IPAdapterStyleCompositionBatch",
    "IPAdapterFaceID", "IPAAdapterFaceIDBatch", "IPAdapterFaceIDKolors",
    "IPAdapterEmbeds", "IPAdapterEmbedsBatch", "IPAdapterEncoder", "IPAdapterCombineEmbeds",
    "PrepImageForClipVision",
}


class NodeMetadataTests(unittest.TestCase):
    def test_help_covers_inputs_and_matches_declared_outputs(self):
        path = Path(__file__).resolve().parents[1] / "IPAdapterPlus.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        definitions = [node for node in tree.body if (
            isinstance(node, ast.ClassDef) and node.name in NODE_NAMES
        ) or (
            isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "WEIGHT_TYPES" for target in node.targets)
        )]
        namespace = {"folder_paths": SimpleNamespace(get_filename_list=lambda category: ["example.bin"])}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
        for name in sorted(NODE_NAMES):
            node = namespace[name]
            with self.subTest(node=name):
                self.assertTrue(node.DESCRIPTION.strip())
                self.assertEqual(len(node.OUTPUT_TOOLTIPS), len(node.RETURN_TYPES))
                self.assertTrue(all(tooltip.strip() for tooltip in node.OUTPUT_TOOLTIPS))
                for group in node.INPUT_TYPES().values():
                    for input_name, schema in group.items():
                        with self.subTest(input=input_name):
                            self.assertIsInstance(schema[1], dict)
                            self.assertTrue(schema[1]["tooltip"].strip())

    def test_optional_composition_uses_style_and_explicit_composition_is_preserved(self):
        path = Path(__file__).resolve().parents[1] / "IPAdapterPlus.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        names = {"IPAdapterAdvanced", "IPAdapterStyleComposition", "IPAdapterStyleCompositionBatch"}
        definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        sdxl = type("SDXL", (), {})
        work_model = object()
        backend = Mock(return_value=(work_model, None))
        namespace = {
            "WEIGHT_TYPES": [], "ipadapter_execute": backend,
            "comfy": SimpleNamespace(model_base=SimpleNamespace(SDXL=sdxl, SDXLRefiner=sdxl, SDXL_instructpix2pix=sdxl)),
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
        model = SimpleNamespace(model=sdxl(), clone=Mock(return_value=work_model))
        bundle = {"ipadapter": {"model": object()}, "clipvision": {"model": object()}}
        style, composition = object(), object()
        for name in ("IPAdapterStyleComposition", "IPAdapterStyleCompositionBatch"):
            node = namespace[name]()
            schema = node.INPUT_TYPES()
            self.assertNotIn("image_composition", schema["required"])
            self.assertIn("image_composition", schema["optional"])
            self.assertTrue({"model", "ipadapter", "image_style"} <= schema["required"].keys())
            for explicit in (False, True):
                with self.subTest(node=name, explicit=explicit):
                    backend.reset_mock()
                    args = {"image_composition": composition} if explicit else {}
                    node.apply_ipadapter(model, bundle, image_style=style, **args)
                    backend.assert_called_once()
                    self.assertIs(backend.call_args.kwargs["image"], style)
                    self.assertIs(backend.call_args.kwargs["image_composition"], composition if explicit else style)
                    self.assertEqual(backend.call_args.kwargs["unfold_batch"], name.endswith("Batch"))


if __name__ == "__main__":
    unittest.main()
