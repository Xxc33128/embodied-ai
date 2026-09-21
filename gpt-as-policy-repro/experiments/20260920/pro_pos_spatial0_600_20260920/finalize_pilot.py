"""Finalize each completed pilot and request service restoration after all finish."""
import csv
import json
import time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

base = Path('/workspace/data/pro_pos_spatial0_600_20260920')
methods = ('student', 'direct', 'hybrid')
processed = set()
deadline = time.monotonic() + 3600
while time.monotonic() < deadline:
    for method in methods:
        folder = base / method
        if method in processed or not (folder / 'result.json').exists():
            continue
        try:
            json.loads((folder / 'result.json').read_text())
        except ValueError:
            continue
        try:
            with imageio.get_writer(str(folder / 'episode.mp4'), fps=20) as writer:
                for path in sorted((folder / 'frames').glob('*_agentview.png')):
                    wrist = path.with_name(path.name.replace('_agentview', '_wrist'))
                    writer.append_data(np.concatenate([imageio.imread(path), imageio.imread(wrist)], axis=1))
            (folder / 'video_note.txt').write_text('Left: agentview. Right: wrist. Playback 20 frames/sec; model waiting excluded.\n')
        except Exception as error:
            (folder / 'video_error.txt').write_text(repr(error))
        processed.add(method)
    if len(processed) == 3:
        break
    time.sleep(5)
rows = []
for method in methods:
    path = base / method / 'result.json'
    if path.exists():
        result = json.loads(path.read_text())
        record = result.get('record', {})
        rows.append(dict(method=method, status=result['status'], success=record.get('success'),
                         steps=record.get('done_step'), wall_seconds=result.get('wall_seconds'),
                         error=record.get('infra_error')))
with (base / 'summary.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['method', 'status', 'success', 'steps', 'wall_seconds', 'error'])
    writer.writeheader()
    writer.writerows(rows)
(base / 'STOP_SERVICES').touch()
(base / 'FINALIZED').write_text(json.dumps({'completed_methods': sorted(processed)}))
