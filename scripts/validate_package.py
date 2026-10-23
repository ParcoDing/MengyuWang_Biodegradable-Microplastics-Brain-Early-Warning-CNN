"""Validate packaged inputs without fitting models or modifying results."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = np.load(ROOT / 'data/features/feature_names.npy', allow_pickle=False)
    assert names.shape == (16,) and len(set(names.tolist())) == 16
    with (ROOT / 'data/sample_manifest.csv').open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    expected = {'day5': [336, 336, 356, 364], 'day14': [404, 508, 368, 296]}
    groups = set()
    for day, counts in expected.items():
        for dose, count in zip(['0CT', '50ppb', '200ppb', '500ppb'], counts):
            a = np.load(ROOT / f'data/features/{day}_{dose}_arr.npy', allow_pickle=False)
            b = np.load(ROOT / f'results/cnn/{day}_{dose}_deep_learning_logit.npy', allow_pickle=False)
            assert a.shape == (16, count), (day, dose, a.shape)
            assert b.shape == (count,), (day, dose, b.shape)
            assert np.isfinite(b).all(), (day, dose, 'nonfinite logits')
    for row in rows:
        p = ROOT / row['image_path']
        header = p.read_bytes()[:24]
        assert header[:8] == b'\x89PNG\r\n\x1a\n'
        assert struct.unpack('>II', header[16:24]) == (256, 256), str(p)
        mask = np.load(ROOT / row['segmentation_path'], allow_pickle=False)
        assert mask.shape == (256, 256) and np.issubdtype(mask.dtype, np.integer)
        assert mask.min() >= 0
        groups.add(row['group_id'])
    assert len(rows) == 2968 and len(groups) == 742
    assert len(list((ROOT / 'data/images').rglob('*.png'))) == 2968
    assert len(list((ROOT / 'data/segmentation').rglob('*_labels.npy'))) == 2968
    for p in (ROOT / 'scripts').glob('*.py'):
        ast.parse(p.read_text(encoding='utf8'), filename=str(p))
    assert not any(p.stat().st_size >= 100 * 1024**2 for p in ROOT.rglob('*') if p.is_file())
    checksum_file = ROOT / 'docs/checksums.sha256'
    if checksum_file.exists():
        for line in checksum_file.read_text(encoding='utf8').splitlines():
            digest, rel = line.split('  ', 1)
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    print(json.dumps({'status': 'PASS', 'images': len(rows), 'fish_groups': len(groups),
                      'features': len(names), 'cnn_training_verified': False}, indent=2))


if __name__ == '__main__':
    main()
