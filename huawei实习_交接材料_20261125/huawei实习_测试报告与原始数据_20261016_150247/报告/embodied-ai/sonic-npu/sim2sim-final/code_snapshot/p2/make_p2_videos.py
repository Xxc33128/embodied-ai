#!/usr/bin/env python3
import pathlib, cv2, numpy as np
P2Root = pathlib.Path(r"E:/sim2sim-week-2026-08-26/12_p2_arm_only")
isaac_path = P2Root / "00_inputs/isaac_reference/video.mp4"
native_path = pathlib.Path(r"E:/sim2sim-week-2026-08-26/10_p1_fullbody_elbow/03_runs/mj_native_r1/video.mp4")
elbow_path = pathlib.Path(r"E:/sim2sim-week-2026-08-26/10_p1_fullbody_elbow/03_runs/mj_matched_r1/video.mp4")
arm_path = P2Root / "03_runs/mj_arm_r1/video.mp4"

out1 = P2Root / "05_media/isaac_native_10s_labeled.mp4"
out2 = P2Root / "05_media/elbow_only_arm_only_10s_labeled.mp4"

def load_frames(path):
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps==0: fps=30.0
    frames=[]
    while True:
        ret,frame=cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames, fps

isaac_frames, isaac_fps = load_frames(isaac_path)
native_frames, native_fps = load_frames(native_path)
elbow_frames, elbow_fps = load_frames(elbow_path)
arm_frames, arm_fps = load_frames(arm_path)
print(f"isaac {len(isaac_frames)} fps {isaac_fps}, native {len(native_frames)} fps {native_fps}, elbow {len(elbow_frames)} fps {elbow_fps}, arm {len(arm_frames)} fps {arm_fps}")

# target fps 30, duration 10s => 300 frames, but MuJoCo videos are 295 frames ~ 9.83s at 30fps, Isaac 250 ~8.33s
# We'll create 295 frames at 30fps (as P1 did), using 295 as target
target_frames = 295
target_fps = 30.0
height = 480
# Resize each frame to height 480, width proportional? Original frames are likely 1280x720 or 1920x1080? Let's check first frame shape
print("isaac shape", isaac_frames[0].shape if isaac_frames else None)
print("native shape", native_frames[0].shape if native_frames else None)
# We'll resize to 640x480 for each panel? For 2 panels, total width 1280
# For P1 they used 1920x390 for 3 panels, so each ~640 width. We'll use 640x480.
panel_w, panel_h = 640, 480

def resize_frames(frames, target_n):
    # resample to target_n by nearest neighbor in time
    src_n = len(frames)
    resized=[]
    for i in range(target_n):
        src_idx = int(i * src_n / target_n)
        src_idx = min(src_idx, src_n-1)
        frame = frames[src_idx]
        frame = cv2.resize(frame, (panel_w, panel_h))
        resized.append(frame)
    return resized

isaac_rs = resize_frames(isaac_frames, target_frames)
native_rs = resize_frames(native_frames, target_frames)
elbow_rs = resize_frames(elbow_frames, target_frames)
arm_rs = resize_frames(arm_frames, target_frames)

def make_labeled_video(frames_list, labels, out_path, title=None):
    # frames_list: list of list of frames (each panel)
    # labels: list of strings per panel
    n_panels = len(frames_list)
    total_w = panel_w * n_panels
    total_h = panel_h + 30  # extra for title bar
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, target_fps, (total_w, total_h))
    for i in range(target_frames):
        # create canvas
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        # title bar
        canvas[0:30, :] = (30,30,30)
        if title:
            cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        t = i * (10.0 / target_frames)
        # time label on title bar right
        cv2.putText(canvas, f"t={t:4.1f}s", (total_w-120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        for p, frames in enumerate(frames_list):
            x0 = p * panel_w
            canvas[30:30+panel_h, x0:x0+panel_w] = frames[i]
            # label per panel at bottom
            label = labels[p]
            # semi-transparent bar
            cv2.rectangle(canvas, (x0, 30+panel_h-25), (x0+panel_w, 30+panel_h), (0,0,0), -1)
            cv2.putText(canvas, label, (x0+10, 30+panel_h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        out.write(canvas)
    out.release()
    print(f"wrote {out_path} {target_frames} frames {total_w}x{total_h}")

# Video 1: Isaac + Native
make_labeled_video([isaac_rs, native_rs], ["Isaac PhysX dt0.005 dec4", "MuJoCo native armature 0.0685/0.03 dt0.002 ppc10"], out1, title="P2: Isaac vs Native (10s, 50Hz control)")
# Video 2: Elbow vs Arm
make_labeled_video([elbow_rs, arm_rs], ["MuJoCo elbow-only 0.01 (1 joint)", "MuJoCo arm-only 0.01 (14 joints)"], out2, title="P2: Elbow-only vs Arm-only (10s)")

# Also make 4-panel if needed (maybe too small but try)
out4 = P2Root / "05_media/isaac_native_elbow_arm_10s_labeled.mp4"
# 4 panels: each 480x270? To keep height 480, width 320 each => total 1280
panel_w4, panel_h4 = 320, 240
# Need to resize again for 4 panels to smaller
def resize_frames_small(frames, target_n, w, h):
    src_n=len(frames)
    res=[]
    for i in range(target_n):
        src_idx=int(i*src_n/target_n)
        src_idx=min(src_idx, src_n-1)
        frame=cv2.resize(frames[src_idx], (w,h))
        res.append(frame)
    return res

isaac_s = resize_frames_small(isaac_frames, target_frames, panel_w4, panel_h4)
native_s = resize_frames_small(native_frames, target_frames, panel_w4, panel_h4)
elbow_s = resize_frames_small(elbow_frames, target_frames, panel_w4, panel_h4)
arm_s = resize_frames_small(arm_frames, target_frames, panel_w4, panel_h4)
total_w4 = panel_w4*4
total_h4 = panel_h4+30
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(out4), fourcc, target_fps, (total_w4, total_h4))
for i in range(target_frames):
    canvas=np.zeros((total_h4, total_w4, 3), dtype=np.uint8)
    canvas[0:30,:]=(30,30,30)
    cv2.putText(canvas, "P2 4-panel: Isaac | Native | Elbow-only | Arm-only (10s)", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    t=i*(10.0/target_frames)
    cv2.putText(canvas, f"t={t:4.1f}s", (total_w4-100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, cv2.LINE_AA)
    frames=[isaac_s[i], native_s[i], elbow_s[i], arm_s[i]]
    labels=["Isaac","Native","Elbow","Arm"]
    for p, fr in enumerate(frames):
        x0=p*panel_w4
        canvas[30:30+panel_h4, x0:x0+panel_w4]=fr
        cv2.rectangle(canvas, (x0,30+panel_h4-18),(x0+panel_w4,30+panel_h4),(0,0,0),-1)
        cv2.putText(canvas, labels[p], (x0+5,30+panel_h4-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1, cv2.LINE_AA)
    out.write(canvas)
out.release()
print(f"wrote 4-panel {out4}")
