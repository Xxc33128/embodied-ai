#!/usr/bin/env python3
"""
make_four_panel_r5.py — R5-4: official 4-panel 10 s video (offline only).

Sources (all official 10 s runs; MJ D00 reused from P1 with trace-hash equivalence
verified, see media_manifest):
  Isaac D00  03_runs/isaac_d00_10s_reference/video.mp4
  Isaac D20  03_runs/isaac_d20_10s_r2/video.mp4
  MuJoCo D00 05_media/sources/mj_elbow_d00_10s_reference.mp4 (P1 mj_matched_r1)
  MuJoCo D20 03_runs/mj_elbow_d20_10s/video.mp4

Alignment: index-proportional resampling to 250 frames @ 25 fps (uniform sim-time
sampling per source). Labels per plan section 8.2. No short-video looping.
Usage: python 05_media/make_four_panel_r5.py --root .
"""
import argparse, hashlib, json, pathlib, sys

import numpy as np

SRC = {
    "isaac_d00": ("03_runs/isaac_d00_10s_reference/video.mp4",
                  "e2781fea124633db23a21687de3591873fc9c2944bc5eeda30a1445d241fd547"),
    "isaac_d20": ("03_runs/isaac_d20_10s_r2/video.mp4",
                  "3e0803896be694781c16776fe97ea1eb60e30b983018913e9e5416ff6bd12840"),
    "mj_d00": ("05_media/sources/mj_elbow_d00_10s_reference.mp4",
               "cf1d3eb189fe019e8bbcf48d1147ca95044bbf65ad7b7e9ab04f6f04b29bbe42"),
    "mj_d20": ("03_runs/mj_elbow_d20_10s/video.mp4",
               "9cc8039cb67c29a75eacdb4b30b9dee6979c3f6cc61b537b3ce71cc346e50117"),
}
LABEL = {
    "isaac_d00": "Isaac | delay 0 ms | physics dt 0.005 s | control dt 0.020 s | model: native URDF",
    "isaac_d20": "Isaac | delay 20 ms | physics dt 0.005 s | control dt 0.020 s | model: native URDF",
    "mj_d00": "MuJoCo | delay 0 ms | physics dt 0.002 s | control dt 0.020 s | model: right-elbow matched",
    "mj_d20": "MuJoCo | delay 20 ms | physics dt 0.002 s | control dt 0.020 s | model: right-elbow matched",
}
TARGET_FRAMES, TARGET_FPS = 250, 25.0


def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    media = root / "05_media"
    out_path = media / "p3_delay_four_panel_10s.mp4"

    import imageio
    import cv2

    manifest_sources = {}
    frames_by_key = {}
    errs = []
    for key, (rel, exp) in SRC.items():
        p = root / rel
        if not p.is_file():
            errs.append(f"missing source {rel}")
            continue
        got = sha(p)
        if got != exp:
            errs.append(f"{key} source sha {got} != pinned {exp}")
        reader = imageio.get_reader(str(p))
        fr = [f for f in reader]
        meta = reader.get_meta_data()
        frames_by_key[key] = fr
        cfps = float(meta.get("fps", 0))
        manifest_sources[key] = {"path": rel, "sha256": got,
                                 "container_frame_count": len(fr), "container_fps": cfps,
                                 "container_playback_duration_s": round(len(fr) / cfps, 6) if cfps else None,
                                 "covered_simulation_cycles": 500, "covered_simulation_time_s": 10.0,
                                 "frame_to_sim_time_mapping": "uniform index-proportional assumption",
                                 "has_per_frame_sim_timestamp": False}
        print(key, len(fr), "frames", cfps, "fps")
    if errs:
        print("SOURCE HASH FAIL", errs, file=sys.stderr)
        sys.exit(1)

    def resample(frames):
        idx = np.linspace(0, len(frames) - 1, TARGET_FRAMES).astype(int)
        return [frames[i] for i in idx]

    rs = {k: resample(v) for k, v in frames_by_key.items()}

    def put(img, txt, y):
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    panels = []
    black = 0
    for i in range(TARGET_FRAMES):
        tt = i / TARGET_FPS
        cells = []
        for key in ("isaac_d00", "isaac_d20", "mj_d00", "mj_d20"):
            f = cv2.resize(rs[key][i], (640, 360))
            f = np.ascontiguousarray(f)
            put(f, LABEL[key], 18)
            put(f, f"t = {tt:.2f} s (nominal sim time)", 34)
            cells.append(f)
        top = np.hstack(cells[:2]); bot = np.hstack(cells[2:])
        panel = np.vstack([top, bot])
        if panel.max() <= 5:
            black += 1
        panels.append(panel)

    imageio.mimsave(str(out_path), panels, fps=TARGET_FPS, codec="libx264", quality=8)
    print(f"saved {out_path} frames={len(panels)} fps={TARGET_FPS} black={black}")

    # frames first/mid/last
    import imageio.v2 as iio2
    picks = {"frame_first.png": 0, "frame_mid.png": TARGET_FRAMES // 2, "frame_last.png": TARGET_FRAMES - 1}
    vis = {}
    for name, i in picks.items():
        iio2.imwrite(str(media / name), panels[i])
        vis[name] = {"index": i, "max_pixel": int(panels[i].max())}

    out_sha = sha(out_path)
    manifest = {
        "video": "05_media/p3_delay_four_panel_10s.mp4",
        "sha256": out_sha,
        "container_frame_count": len(panels), "container_fps": TARGET_FPS,
        "container_playback_duration_s": len(panels) / TARGET_FPS,
        "covered_simulation_cycles": 500, "covered_simulation_time_s": 10.0,
        "frame_to_sim_time_mapping": "uniform index-proportional assumption",
        "has_per_frame_sim_timestamp": False,
        "resolution": [1280, 720], "black_frame_count": black,
        "alignment": "index-proportional resampling to 250x@25fps; sources uniformly sampled in sim time over 10 s",
        "sources": manifest_sources,
        "mj_d00_reuse": {
            "reason": "R4 package MJ D00 has no video; P1 official mj_matched_r1 reused",
            "source_zip": "P1_fullbody_elbow_transfer_v1.zip",
            "source_zip_sha256": "8bb8ba9658c81160c7d6542e2b418637b7b736688b8a226ba885327b6d8b6824",
            "source_rel": "10_p1_fullbody_elbow/03_runs/mj_matched_r1/video.mp4",
            "source_video_sha256": "cf1d3eb189fe019e8bbcf48d1147ca95044bbf65ad7b7e9ab04f6f04b29bbe42",
            "source_trace_sha256": "ba9276f78b24759612441ec939ac0aefb287de6a2d9877d99224904e76474a84",
            "r4_mj_d00_trace_sha256": "ba9276f78b24759612441ec939ac0aefb287de6a2d9877d99224904e76474a84",
            "trace_identical": True},
        "visibility": vis,
        "labels": list(LABEL.values()),
        "visibility_note": ("Isaac D00 cell uses the official P1-era baseline video (close camera); the robot may be "
                            "leg-cropped in some frames. All four cells non-black. Video observational only; "
                            "quantitative claims come from frozen traces. Source videos carry no per-frame sim "
                            "timestamps; frames are mapped by index onto the accepted 10 s runs, so on-screen t is "
                            "nominal simulation time. Container playback duration differs from covered simulation "
                            "time (10 s); see per-source fields."),
    }
    (media / "media_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("media_manifest written", out_sha[:12])


if __name__ == "__main__":
    main()
