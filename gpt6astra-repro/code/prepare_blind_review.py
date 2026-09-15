"""Build anonymous offline evidence packets, excluding policy notes and auto grades."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import uuid

from PIL import Image, ImageDraw
from audit_standard_trials import valid_trial_folders


def prepare(root):
    packets = root / 'blind-review'
    packets.mkdir(exist_ok=True)
    mapping_path = root / 'blind-review-map.json'
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    for trial in valid_trial_folders(root):
        if trial.name in mapping or not (trial / 'result.json').exists():
            continue
        # A finished result is written before the encoder closes: require final PNGs.
        if len(list(trial.glob('final-*.png'))) != 3:
            continue
        rows = [json.loads(x) for x in (trial / 'physics.jsonl').read_text().splitlines()]
        name = 'clip-' + uuid.uuid4().hex[:10]
        dest = packets / name
        dest.mkdir()
        shutil.copy2(trial / 'video.mp4', dest / 'video.mp4')
        selected = set(range(0, len(rows), 20)) | {len(rows) - 1}
        for k in range(1, len(rows)):
            if any(rows[k][field] != rows[k - 1][field] for field in ['grasped', 'any_gripper_contact']):
                selected.update([k - 1, k])
        selected.add(max(range(len(rows)), key=lambda k: rows[k]['cube_bottom_z']))
        selected.add(min(range(len(rows)), key=lambda k: sum((rows[k]['cube_pos'][d] - rows[k]['bowl_xy'][d]) ** 2 for d in [0, 1])))
        clean = []
        for row in rows:
            item = {key: row[key] for key in ['step', 'cube_pos', 'cube_quat_wxyz', 'cube_bottom_z', 'cube_velocity', 'grasped', 'any_gripper_contact', 'bowl_xy']}
            item['eef_pos'] = row['state']['eef_pos']
            item['measured_gripper'] = row['state']['gripper']
            clean.append(item)
        (dest / 'trajectory.json').write_text(json.dumps(clean, separators=(',', ':')) + '\n')
        frames = []
        for step in sorted(selected):
            target = dest / f'frame-{step:04d}.png'
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(dest / 'video.mp4'), '-vf', f'select=eq(n\\,{step})', '-frames:v', '1', str(target)], check=True)
            frames.append((step, target))
        for start in range(0, len(frames), 6):
            sheet = Image.new('RGB', (1344, 3 * 248), 'white')
            draw = ImageDraw.Draw(sheet)
            for offset, (step, file) in enumerate(frames[start:start + 6]):
                x, y = offset % 2 * 672, offset // 2 * 248
                sheet.paste(Image.open(file), (x, y + 24))
                draw.text((x + 6, y + 5), f'step {step} | {step / 20:.2f}s | front / side / wrist', fill='black')
            sheet.save(dest / f'sheet-{start // 6 + 1:02d}.png')
        (dest / 'evidence.json').write_text(json.dumps({'clip': name, 'control_hz': 20, 'table_top_z': 0.8, 'bowl_xy': rows[0]['bowl_xy'], 'steps': len(rows) - 1, 'selected_frames': sorted(selected), 'identity_and_policy_notes_removed': True}, indent=2) + '\n')
        mapping[trial.name] = name
        mapping_path.write_text(json.dumps(mapping, indent=2) + '\n')
        print(name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    prepare(parser.parse_args().root.resolve())
