import pathlib
import numpy as np
try:
    import imageio.v2 as iio
except: import imageio as iio
import cv2

def add_label(img, text, pos=(30,40)):
    # img is RGB
    out = img.copy()
    # Convert to BGR for cv2
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    # Add black outline for readability
    cv2.putText(bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def make_sidebyside_labeled(isaac_path, mj_path, out_path, isaac_label, mj_label):
    print(f"Making {out_path} ...")
    r1 = iio.get_reader(str(isaac_path))
    r2 = iio.get_reader(str(mj_path))
    # Get fps
    try:
        fps = r1.get_meta_data().get("fps", 30)
    except: fps = 30
    frames = []
    for im1, im2 in zip(r1, r2):
        # Ensure same height
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        if h1 != h2:
            # Resize to 720
            im1 = cv2.resize(im1, (1280,720))
            im2 = cv2.resize(im2, (1280,720))
            h1, w1 = im1.shape[:2]
        # Add labels
        im1_l = add_label(im1, isaac_label, pos=(20,50))
        im2_l = add_label(im2, mj_label, pos=(20,50))
        combined = np.hstack([im1_l, im2_l])
        frames.append(combined)
    r1.close(); r2.close()
    if not frames:
        print("  no frames")
        return
    # Resize combined to 2560x720 if needed (already)
    writer = iio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  wrote {len(frames)} frames to {out_path}")

if __name__ == "__main__":
    base = pathlib.Path("E:/sim2sim-week-2026-08-26/02_lab_dance9")
    isaac_label = "Isaac/PhysX | dt=0.005 | control=0.02 | delay=0 | gain=1.0"
    mj_label = "MuJoCo     | dt=0.002 | control=0.02 | delay=0 | gain=1.0"
    for tag in ["5s", "10s"]:
        isaac = base / f"isaac_{tag}" / "video.mp4"
        mj = base / f"mj_{tag}" / "video.mp4"
        out = base / f"sidebyside_{tag}_labeled.mp4"
        if isaac.is_file() and mj.is_file():
            make_sidebyside_labeled(isaac, mj, out, isaac_label, mj_label)
        else:
            print(f"missing {tag}")
    # Also keep old unlabeled as fallback
