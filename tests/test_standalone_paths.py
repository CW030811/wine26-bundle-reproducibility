"""Check standalone default path expressions without loading models or solvers."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILES = {name for archive in json.loads((ROOT / 'bundles/manifest.json').read_text())['archives'].values()
         for name in archive['files']}
EVALUATORS = ('test_FCP.py', 'test_FCP_log.py', 'test_FCP_f33.py', 'test_PCP_cp.py')


def entrypoint(name):
    path = ROOT / 'src/deterministic' / name
    tree = ast.parse(path.read_text())
    main = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main'), None)
    body = main.body if main else tree.body[-1].body
    return path, ast.Module(body=body, type_ignores=[])


def expression(module, name):
    return next(node.value for node in ast.walk(module) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def evaluate(node, scope):
    return eval(compile(ast.Expression(body=node), '<standalone-path>', 'eval'), scope)


def defaults(name):
    path, module = entrypoint(name)
    scope = {'os': os, '__file__': str(path)}
    scope['dir_path'] = evaluate(expression(module, 'dir_path'), scope)
    return module, scope


class StandalonePathTests(unittest.TestCase):
    def test_evaluator_checkpoints_are_packaged(self):
        for name in EVALUATORS:
            with self.subTest(source=name):
                module, scope = defaults(name)
                path = Path(evaluate(expression(module, 'model_path'), scope))
                self.assertTrue(path.relative_to(ROOT).as_posix() in FILES, str(path))

    def test_evaluator_input_directories_are_packaged(self):
        for name in EVALUATORS:
            with self.subTest(source=name):
                module, scope = defaults(name)
                path = Path(evaluate(expression(module, 'test_data_path'), scope))
                prefix = path.relative_to(ROOT).as_posix() + '/'
                self.assertTrue(any(name.startswith(prefix) for name in FILES), prefix)

    def test_fcp_sample_path_join_preserves_the_input_directory(self):
        for name in EVALUATORS[:3]:
            with self.subTest(source=name):
                module, scope = defaults(name)
                directory = evaluate(expression(module, 'test_data_path'), scope)
                prefix = Path(directory).relative_to(ROOT).as_posix() + '/'
                sample = next(name for name in sorted(FILES) if name.startswith(prefix))
                scope.update(test_data_path=directory, dir_list=[Path(sample).name], i=0)
                actual = Path(evaluate(expression(module, 'file_path'), scope))
                self.assertEqual(actual, ROOT / sample)

    def test_pcp_generator_checkpoint_directory_is_packaged(self):
        module, scope = defaults('generate_data_PCP_cp.py')
        path = Path(evaluate(expression(module, 'model_dir'), scope))
        prefix = path.relative_to(ROOT).as_posix() + '/'
        self.assertTrue(any(name.startswith(prefix) for name in FILES), prefix)

    def test_pcp_generator_outputs_go_under_results(self):
        module, scope = defaults('generate_data_PCP_cp.py')
        call = next(node for node in ast.walk(module) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'generate_sample')
        scope['args'] = SimpleNamespace(folder_name='test_generation')
        path = Path(evaluate(call.args[4], scope))
        self.assertTrue(path.is_relative_to(ROOT / 'results'), str(path))


if __name__ == '__main__':
    unittest.main()
