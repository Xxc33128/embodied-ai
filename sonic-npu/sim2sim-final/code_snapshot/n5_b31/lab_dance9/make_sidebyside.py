import pathlib, sys
import numpy as np

try:
    import imageio
    has_imageio=True
except: has_imageio=False
try:
    import cv2
    has_cv2=True
except: has_cv2=False

def make_sidebyside(isaac_path, mj_path, out_path):
    print(f"Making side-by-side: {isaac_path} + {mj_path} -> {out_path}")
    if has_imageio:
        try:
            r1 = imageio.get_reader(str(isaac_path))
            r2 = imageio.get_reader(str(mj_path))
            # Get fps and size
            fps1 = r1.get_meta_data().get("fps", 30)
            fps2 = r2.get_meta_data().get("fps", 30)
            fps = min(fps1, fps2)
            # Collect frames
            frames = []
            # Use the shorter video length
            for im1, im2 in zip(r1, r2):
                # Resize if needed to same height
                h1, w1 = im1.shape[:2]
                h2, w2 = im2.shape[:2]
                # Both should be 720p, but ensure same height
                if h1 != h2:
                    # resize via cv2 if available
                    if has_cv2:
                        im2 = cv2.resize(im2, (w1, h1))
                    else:
                        # crop/pad
                        min_h = min(h1, h2)
                        im1 = im1[:min_h]
                        im2 = im2[:min_h]
                # hstack
                combined = np.hstack([im1, im2])
                frames.append(combined)
            if not frames:
                print("  no frames")
                return
            writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
            for f in frames:
                writer.append_data(f)
            writer.close()
            print(f"  wrote {len(frames)} frames to {out_path} via imageio")
            return
        except Exception as e:
            print(f"  imageio failed: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
    if has_cv2:
        try:
            cap1 = cv2.VideoCapture(str(isaac_path))
            cap2 = cv2.VideoCapture(str(mj_path))
            fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30
            fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30
            fps = min(fps1, fps2)
            w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)); h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)); h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_w = w1 + w2
            out_h = max(h1, h2)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
            count=0
            while True:
                ret1, f1 = cap1.read()
                ret2, f2 = cap2.read()
                if not ret1 or not ret2: break
                if f1.shape[0] != out_h or f2.shape[0] != out_h:
                    f1 = cv2.resize(f1, (w1, out_h))
                    f2 = cv2.resize(f2, (w2, out_h))
                combined = np.hstack([f1, f2])
                # Convert BGR? cv2 reads BGR, writer expects BGR, so keep
                out.write(combined)
                count+=1
            cap1.release(); cap2.release(); out.release()
            print(f"  wrote {count} frames via cv2")
            return
        except Exception as e:
            print(f"  cv2 failed: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
    print("  FAILED to create side-by-side: no backend")

if __name__=="__main__":
    base = pathlib.Path("E:/sim2sim-week-2026-08-26/02_lab_dance9")
    for tag in ["5s", "10s"]:
        isaac = base / f"isaac_{tag}" / "video.mp4"
        mj = base / f"mj_{tag}" / "video.mp4"
        out = base / f"sidebyside_{tag}.mp4"
        if isaac.is_file() and mj.is_file():
            make_sidebyside(isaac, mj, out)
        else:
            print(f"  missing {tag}: isaac {isaac.is_file()} mj {mj.is_file()}")
