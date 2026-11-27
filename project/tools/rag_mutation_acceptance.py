"""Prove tests detect deliberate defects in disposable source copies only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from rag_local import PROJECT


def hashes():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (PROJECT / 'src').rglob('*.py')}


def run(output):
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh evidence directory')
    out.mkdir(parents=True, exist_ok=True)
    before = hashes()
    result = {'checks': {}, 'mutations': {}}
    mutations = [
        ('plugin_contract', 'src/rag_multidocs/rerankers/registry.py',
         'type(r.score) not in (float, int) or not math.isfinite(r.score) or not 0 <= r.score <= 1',
         'False  # Deliberate isolated nonfinite/out-of-range score acceptance defect',
         'tests/test_a_plus_plugins.py::test_invalid_plugins_fall_back_to_exact_rrf'),
        ('coverage_algorithm', 'src/rag_multidocs/retrieval/controller.py',
         'policy.gamma * gain', '0 * gain',
         'tests/test_p1_adaptive.py::test_hand_computable_coverage_gain_changes_second_choice'),
    ]
    with tempfile.TemporaryDirectory(prefix='rag-p1-mutation-') as temporary:
        copy = Path(temporary) / 'project'
        copy.mkdir()
        for folder in ['src', 'tests', 'connectors', 'rerankers']:
            shutil.copytree(PROJECT / folder, copy / folder,
                            ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.pyc'))
        shutil.copyfile(PROJECT / 'pyproject.toml', copy / 'pyproject.toml')
        env = dict(os.environ, PYTHONPATH=str(copy / 'src'),
                   PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for name, relative, original, defect, target in mutations:
            path = copy / relative
            text = path.read_text()
            if text.count(original) != 1:
                raise ValueError('Mutation target drifted: ' + relative)
            codes = []
            for phase in ['before', 'mutated', 'restored']:
                path.write_text(text.replace(original, defect) if phase == 'mutated' else text)
                command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', target]
                test = subprocess.run(command, cwd=copy, env=env, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                (out / f'{name}_{phase}.log').write_text(test.stdout)
                codes.append(test.returncode)
            result['mutations'][name] = {'target': target, 'file': relative, 'returncodes': codes}
            result['checks'][name] = codes[0] == 0 and codes[1] == 1 and codes[2] == 0
    result['core_before'] = before
    result['core_after'] = hashes()
    result['checks']['live_source_unchanged'] = bool(before) and before == result['core_after']
    result['status'] = 'PASS' if all(result['checks'].values()) else 'FAIL'
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
