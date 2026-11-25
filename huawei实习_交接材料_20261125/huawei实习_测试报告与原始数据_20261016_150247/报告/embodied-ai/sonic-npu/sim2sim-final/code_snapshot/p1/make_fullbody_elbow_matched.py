#!/usr/bin/env python3
"""
make_fullbody_elbow_matched.py — P1 single-variable armature injection

Rules (P1 §6.1-6.2):
- native right_elbow_pitch_joint inherits armature 0.0685 via class elbow_pitch (no explicit attribute)
- matched MUST insert explicit armature="0.01" on THAT joint tag only
- MUST NOT modify <default class="elbow_pitch"> nor left elbow nor any other field
- Diff must contain exactly the target joint line change
- Other text must be byte-identical except that insertion
"""
import difflib
import hashlib
import pathlib
import re
import sys

P1Root = pathlib.Path(__file__).parent.parent
native_xml = pathlib.Path(__file__).parent / "l7_29dof_neck_fixed_native.xml"
matched_xml = pathlib.Path(__file__).parent / "l7_29dof_neck_fixed_elbow_matched.xml"
diff_path = pathlib.Path(__file__).parent / "xml_target_diff.patch"
native_yaml = pathlib.Path(__file__).parent / "mimic_dance_9_native.yaml"
matched_yaml = pathlib.Path(__file__).parent / "mimic_dance_9_elbow_matched.yaml"

def sha256(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    # 1. UTF-8 read native XML
    text = native_xml.read_text(encoding="utf-8")
    # 2. Locate unique right_elbow_pitch_joint joint tag
    # pattern matches <joint ... name="right_elbow_pitch_joint" ... />
    pattern = re.compile(r'<joint[^>]*name="right_elbow_pitch_joint"[^>]*/?>')
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        print(f"FAIL: found {len(matches)} matches for right_elbow_pitch_joint joint tag, expected 1")
        for m in matches:
            print(m.group(0)[:200])
        sys.exit(1)
    m = matches[0]
    tag = m.group(0)
    print(f"Found target joint tag: {tag}")

    # 3. Assert no explicit armature
    if "armature" in tag:
        print(f"FAIL: target tag already has explicit armature: {tag}")
        sys.exit(1)

    # Also ensure default class elbow_pitch unchanged later — we will verify diff
    # 4. Only insert armature="0.01"
    # Insert before class="elbow_pitch" or before /> . Safer: insert before class attribute's preceding space?
    # Original: <joint name="right_elbow_pitch_joint" pos="0 0 0" axis="0 1 0" range="-2.36 0.7" class="elbow_pitch"/>
    # Desired: <joint name="right_elbow_pitch_joint" pos="0 0 0" axis="0 1 0" range="-2.36 0.7" class="elbow_pitch" armature="0.01"/>
    # Do insertion by replacing class="elbow_pitch" with class="elbow_pitch" armature="0.01"
    if 'class="elbow_pitch"' not in tag:
        print(f"FAIL: target tag missing class=\"elbow_pitch\": {tag}")
        sys.exit(1)
    new_tag = tag.replace('class="elbow_pitch"', 'class="elbow_pitch" armature="0.01"')
    # Verify new_tag contains armature
    assert 'armature="0.01"' in new_tag
    # Also verify left elbow NOT touched (should remain without armature)
    if 'left_elbow_pitch_joint' in text:
        left_pattern = re.compile(r'<joint[^>]*name="left_elbow_pitch_joint"[^>]*/?>')
        left_m = list(left_pattern.finditer(text))
        assert len(left_m)==1
        assert 'armature' not in left_m[0].group(0), "left elbow should not have armature"

    # 5. Other text unchanged: build new text by splicing
    new_text = text[:m.start()] + new_tag + text[m.end():]

    # Verify only one occurrence of armature="0.01" in new_text beyond default?
    # Default class has armature 0.0685, not 0.01, so count of 'armature="0.01"' should be 1
    count = new_text.count('armature="0.01"')
    if count != 1:
        print(f"FAIL: new_text has {count} occurrences of armature=\"0.01\", expected 1")
        sys.exit(1)

    # Also ensure left elbow still without armature
    left_after = re.compile(r'<joint[^>]*name="left_elbow_pitch_joint"[^>]*/?>').search(new_text)
    if 'armature' in left_after.group(0):
        print("FAIL: left elbow incorrectly modified")
        sys.exit(1)

    # Write matched XML
    matched_xml.write_text(new_text, encoding="utf-8")
    print(f"Wrote matched XML: {matched_xml} sha256={sha256(matched_xml)}")
    print(f"Native SHA: {sha256(native_xml)}")

    # 6. Output unified diff
    native_lines = text.splitlines(keepends=True)
    matched_lines = new_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(native_lines, matched_lines,
                                     fromfile="l7_29dof_neck_fixed_native.xml",
                                     tofile="l7_29dof_neck_fixed_elbow_matched.xml",
                                     lineterm=""))
    # diff lines already have no lineterm, join with \n
    diff_text = "\n".join(diff) + "\n" if diff else ""
    diff_path.write_text(diff_text, encoding="utf-8")
    print(f"Wrote diff: {diff_path} lines={len(diff)}")
    print(diff_text[:2000])

    # 7. Diff must only contain target joint line
    # Check diff hunk: should have exactly one removed and one added line, both containing right_elbow_pitch_joint
    added = [l for l in diff if l.startswith("+") and not l.startswith("+++")]
    removed = [l for l in diff if l.startswith("-") and not l.startswith("---")]
    # unified diff header includes @@ line
    if len(added) != 1 or len(removed) != 1:
        print(f"FAIL: diff added={len(added)} removed={len(removed)}, expected 1 each")
        print(f"added: {added}")
        print(f"removed: {removed}")
        sys.exit(1)
    if "right_elbow_pitch_joint" not in added[0] or "right_elbow_pitch_joint" not in removed[0]:
        print(f"FAIL: diff lines must contain right_elbow_pitch_joint")
        sys.exit(1)
    if 'armature="0.01"' not in added[0]:
        print(f"FAIL: added line missing armature=\"0.01\"")
        sys.exit(1)
    if 'armature' in removed[0]:
        print(f"FAIL: removed line should not have armature")
        sys.exit(1)
    # Ensure diff does not contain other joint names or default class change
    for bad in ["left_elbow", "default class=\"elbow_pitch\"", "timestep", "hip_pitch"]:
        for l in added + removed:
            if bad in l and "right_elbow_pitch_joint" not in l:
                print(f"FAIL: diff contains unintended change {bad}: {l}")
                sys.exit(1)
    print("Diff check PASS: only target joint line changed")

    # 7b. Matched YAML: copy native, only change xml_path
    yaml_text = native_yaml.read_text(encoding="utf-8")
    # Ensure native has expected xml_path
    if 'l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml' not in yaml_text:
        print("FAIL: native yaml missing expected xml_path")
        sys.exit(1)
    new_yaml_text = yaml_text.replace(
        'xml_path: "l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml"',
        'xml_path: "l7_29dof_neck_fixed/p1_generated/l7_29dof_neck_fixed_elbow_matched.xml"'
    )
    # Ensure only that line changed
    if yaml_text == new_yaml_text:
        print("FAIL: yaml replacement did not change anything")
        sys.exit(1)
    # Verify diff is single line
    import difflib as dl
    ydiff = list(dl.unified_diff(yaml_text.splitlines(), new_yaml_text.splitlines(), lineterm=""))
    yadded = [l for l in ydiff if l.startswith("+") and not l.startswith("+++")]
    yremoved = [l for l in ydiff if l.startswith("-") and not l.startswith("---")]
    if len(yadded)!=1 or len(yremoved)!=1:
        print(f"FAIL yaml diff added {len(yadded)} removed {len(yremoved)}")
        sys.exit(1)
    matched_yaml.write_text(new_yaml_text, encoding="utf-8")
    print(f"Wrote matched YAML: {matched_yaml} sha256={sha256(matched_yaml)}")
    print("YAML diff check PASS")

    print("make_fullbody_elbow_matched.py SUCCESS")

if __name__ == "__main__":
    main()
