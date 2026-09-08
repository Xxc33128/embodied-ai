#!/usr/bin/env python3
"""make_fullbody_arm_matched.py — P2 ARM_ONLY 14 joints armature injection (from native XML 69e975...)"""
import difflib, hashlib, pathlib, re, sys
P2_ASSET = pathlib.Path(__file__).parent
NATIVE_XML = P2_ASSET / "l7_29dof_neck_fixed_native.xml"
MATCHED_XML = P2_ASSET / "l7_29dof_neck_fixed_arm_matched.xml"
DIFF_PATH = P2_ASSET / "xml_arm_target_diff.patch"
NATIVE_YAML = P2_ASSET / "mimic_dance_9_native.yaml"
MATCHED_YAML = P2_ASSET / "mimic_dance_9_arm_matched.yaml"

ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_arm_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_arm_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"
]

def sha256(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    text = NATIVE_XML.read_text(encoding="utf-8")
    new_text = text
    for joint in ARM_JOINTS:
        pat = re.compile(rf'<joint[^>]*name="{re.escape(joint)}"[^>]*/?>')
        ms = list(pat.finditer(new_text))
        if len(ms) != 1:
            print(f"FAIL: {joint} found {len(ms)} matches, expected 1")
            sys.exit(1)
        tag = ms[0].group(0)
        if "armature" in tag:
            print(f"FAIL: {joint} already has armature: {tag}")
            sys.exit(1)
        if 'class="' in tag:
            new_tag = re.sub(r'(class="[^"]+")', r'\1 armature="0.01"', tag, count=1)
        else:
            new_tag = tag.replace("/>", ' armature="0.01"/>')
            if new_tag == tag:
                new_tag = tag.replace(">", ' armature="0.01">')
        assert 'armature="0.01"' in new_tag
        new_text = new_text[:ms[0].start()] + new_tag + new_text[ms[0].end():]
        print(f"Patched {joint}")
    count = new_text.count('armature="0.01"')
    if count != 14:
        print(f"FAIL: new_text has {count} armatures, expected 14")
        sys.exit(1)
    for other in ["left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]:
        pat = re.compile(rf'<joint[^>]*name="{re.escape(other)}"[^>]*/?>')
        m = pat.search(new_text)
        if m and 'armature="0.01"' in m.group(0):
            print(f"FAIL: other joint {other} incorrectly modified")
            sys.exit(1)
    MATCHED_XML.write_text(new_text, encoding="utf-8", newline="\n")
    print(f"Wrote matched {MATCHED_XML} sha {sha256(MATCHED_XML)} native {sha256(NATIVE_XML)}")
    native_lines = text.splitlines(keepends=True)
    matched_lines = new_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(native_lines, matched_lines, fromfile="l7_29dof_neck_fixed_native.xml", tofile="l7_29dof_neck_fixed_arm_matched.xml", lineterm=""))
    diff_text = "\n".join(diff) + "\n" if diff else ""
    DIFF_PATH.write_text(diff_text, encoding="utf-8", newline="\n")
    print(f"Wrote diff {DIFF_PATH} lines {len(diff)}")
    added = [l for l in diff if l.startswith("+") and not l.startswith("+++")]
    removed = [l for l in diff if l.startswith("-") and not l.startswith("---")]
    if len(added) != 14 or len(removed) != 14:
        print(f"FAIL: diff added {len(added)} removed {len(removed)} expected 14 each")
        sys.exit(1)
    for a in added:
        if 'armature="0.01"' not in a:
            print(f"FAIL: added missing armature {a}")
            sys.exit(1)
    print("Diff check PASS 14 lines")
    yaml_text = NATIVE_YAML.read_text(encoding="utf-8")
    if 'l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml' not in yaml_text:
        print("FAIL native yaml missing xml_path")
        sys.exit(1)
    new_yaml = yaml_text.replace('xml_path: "l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml"', 'xml_path: "l7_29dof_neck_fixed/p2_generated/l7_29dof_neck_fixed_arm_matched.xml"')
    if yaml_text == new_yaml:
        print("FAIL yaml not changed")
        sys.exit(1)
    MATCHED_YAML.write_text(new_yaml, encoding="utf-8", newline="\n")
    print(f"Wrote matched yaml {MATCHED_YAML} sha {sha256(MATCHED_YAML)}")
    print("SUCCESS")

if __name__ == "__main__": main()
