"""Offline recording checks; never used to supply policy feedback or grade intent."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess


def audit(root):
    protocol = json.loads((root / 'standard/protocol.json').read_text())
    rows = []
    for folder in sorted(root.glob('trial-[0-9][0-9]')):
        if not (folder / 'result.json').exists():
            continue
        result = json.loads((folder / 'result.json').read_text())
        problems = []
        def check(condition, label):
            if not condition:
                problems.append(label)
        index = result['trial_index']
        initial = json.loads((folder / 'initialization.json').read_text())
        bank = json.loads((root / 'initializations' / folder.name / 'initialization.json').read_text())
        check(initial == bank, 'initial state differs')
        physics = [json.loads(line) for line in (folder / 'physics.jsonl').read_text().splitlines()]
        check([p['step'] for p in physics] == list(range(result['physics_steps'] + 1)), 'step sequence')
        check(result['physics_steps'] <= 900 and result['calls'] <= 20, 'budget exceeded')
        outgoing = sorted(folder.glob('outgoing-*.json'))
        check(len(outgoing) == len(list(folder.glob('incoming-*.json'))) == result['calls'], 'request/response count')
        images = 0
        frame_index = []
        frame_folder = folder / 'frames'
        frame_folder.mkdir(exist_ok=True)
        schemas = set()
        for call, path in enumerate(outgoing, 1):
            request = json.loads(path.read_text())
            check(path.name == f'outgoing-{call:02d}.json', 'call sequence')
            check(hashlib.sha256(request['messages'][0]['content'].encode()).hexdigest() == protocol['assembled_system_sha256'], 'system prompt differs')
            schemas.add(json.dumps(request['tools'], sort_keys=True))
            started = json.loads((folder / f'call-{call:02d}-started.json').read_text())
            ended = json.loads((folder / f'call-{call:02d}-ended.json').read_text())
            check(started['physics_steps'] == ended['physics_steps'], 'simulation ran during inference')
            count = 0
            call_frames = []
            for message in request['messages']:
                # Model-authored notes can contain these words; inspect only observation states.
                content = message.get('content')
                if not isinstance(content, list):
                    continue
                for block in content:
                    if block.get('type') == 'image_url':
                        raw = base64.b64decode(block['image_url']['url'].split(',', 1)[1], validate=True)
                        check(raw.startswith(b'\x89PNG\r\n\x1a\n'), 'invalid PNG')
                        count += 1
                        call_frames.append(raw)
                    elif message['role'] == 'user' and block.get('type') == 'text':
                        for field in protocol['observations']['private_fields_excluded']:
                            check(f'state[{field}]' not in block['text'], 'private observation field ' + field)
            check(count == (3 if call == 1 else 6), 'image horizon or camera count')
            for camera, raw in zip(['front', 'side', 'wrist'], call_frames[-3:]):
                filename = f'call-{call:02d}-{camera}.png'
                (frame_folder / filename).write_bytes(raw)
                frame_index.append({'call': call, 'step': started['physics_steps'], 'camera': camera, 'file': filename, 'sha256': hashlib.sha256(raw).hexdigest()})
            images += count
        (frame_folder / 'index.json').write_text(json.dumps(frame_index, indent=2) + '\n')
        check(len(schemas) == 1, 'tool schemas changed')
        probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames', '-of', 'json', str(folder / 'video.mp4')], capture_output=True, text=True)
        if probe.returncode:
            problems.append('video decode failed: ' + probe.stderr.strip())
            video_frames = None
        else:
            stream = json.loads(probe.stdout)['streams'][0]
            video_frames = int(stream['nb_read_frames'])
            check((stream['width'], stream['height'], stream['r_frame_rate']) == (672, 224, '20/1'), 'video geometry/fps')
            check(video_frames == result['physics_steps'] + 1 == result['frames'], 'video frame count')
        check(bool(result['termination_reasons']), 'missing termination')
        check(len(list(folder.glob('initial-*.png'))) == len(list(folder.glob('final-*.png'))) == 3, 'endpoint images')
        row = {'trial': index + 1, 'scene_seed': 2000 + index, 'env_success': result['metrics'].get('success_at_end'), 'calls': result['calls'], 'physics_steps': result['physics_steps'], 'sim_seconds': result['physics_steps'] / 20, 'wall_seconds': round(result['wall_seconds'], 2), 'video_frames': video_frames, 'image_references': images, 'termination': json.dumps(result['termination_reasons'], ensure_ascii=False), 'recording_checks': 'PASS' if not problems else '; '.join(problems), 'reviewed_stage': '', 'reviewed_success': ''}
        review = folder / 'review.json'
        if review.exists():
            data = json.loads(review.read_text())
            row.update(reviewed_stage=data['stage_max'], reviewed_success=data['stage_max'] == 4)
        rows.append(row)
    (root / 'recording-audit.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    if rows:
        with (root / 'summary.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    cards = []
    for row in rows:
        name = f"trial-{row['trial']:02d}"
        cards.append(f'<article><h2>Trial {row["trial"]:02d} · seed {row["scene_seed"]}</h2><p>{row["calls"]} 次调用 · {row["physics_steps"]} 控制步 · 仿真 {row["sim_seconds"]:.2f}s · 墙钟 {row["wall_seconds"]:.1f}s</p><video controls preload="metadata" src="{name}/video.mp4"></video><p>录像：front / side / wrist。记录检查：{html.escape(row["recording_checks"])}；审核阶段：{row["reviewed_stage"] if row["reviewed_stage"] != "" else "待复核"}</p><a href="{name}/result.json">原始结果</a> · <a href="{name}/physics.jsonl">物理轨迹</a></article>')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Bowl 标准实验记录</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#f5f4f0;color:#202b32}h1{font-size:32px}article{padding:20px 0;border-top:1px solid #bbb}video{width:100%;background:#222}a{color:#176981}p{line-height:1.6}</style>' + f'<h1>Bowl 标准实验记录</h1><p>已完成 {len(rows)}/20 个 trial。GPT‑6 Astra 子agent接入；每次决策新上下文，统一20调用/900控制步。视频为20fps仿真时钟，推理期间仿真暂停。成功须经独立阶段复核，harness执行成功不代表抓放成功。</p>' + ''.join(cards) + '</html>'
    (root / 'index.html').write_text(page)
    print(json.dumps({'completed': len(rows), 'recording_pass': sum(r['recording_checks'] == 'PASS' for r in rows), 'problems': [r for r in rows if r['recording_checks'] != 'PASS']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root.resolve())
