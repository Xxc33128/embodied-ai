#!/bin/bash
# R15 C-line experimental arm: r14 arm B re-run with the FIXED reset-only pin sampler.
# Identical to r14arms.sh arm B except: fixed commands.py (3ab334ce...), resample ledger on.
# Control = existing r14_ctrl_mix (r14arms.sh arm A; p=0 path proven bitwise-identical, S9).
T0=$(date +%s); echo "T0=$(date -Iseconds) epoch=$T0"
cd /data/halo/halo-repo/tracker || exit 1
CK=logs/rsl_rl/motion_tracking_scene_prerl/2026-11-20_10-19-06_labsofa_c1000/model_499.pt
unset MOTION_PIN_START_FRAME MOTION_PIN_START_FRAME_IDX
export MOTION_PIN_START_PROB=0.25
export MOTION_RESAMPLE_STATS=/data/halo/diag/resample_check/r15_exp_stats.json
rm -f "$MOTION_RESAMPLE_STATS"
timeout 50m env OMNI_KIT_ACCEPT_EULA=YES /data/envs/halo-tracker/bin/python training/scripts/rsl_rl/train.py \
  --task Halo-MotionTracking-Robot_Terrain-PreRL --headless \
  --motion_data_folder data/g1_processed/robot-terrain/labsofa --num_envs 64 --max_iterations 400 \
  --load_checkpoint_path "$CK" --resume_optim True --run_name r15_exp_f0mix_fixed \
  > /data/halo/logs/r15_exp.log 2>&1
RC=$?
T1=$(date +%s); echo "TRAIN_RC=$RC TRAIN_SECONDS=$((T1-T0))"
[ "$RC" = "0" ] || { echo EXP-FAIL; exit 1; }
EXP=$(ls -dt logs/rsl_rl/motion_tracking_scene_prerl/*_r15_exp_f0mix_fixed | head -1)/model_399.pt
echo "EXP_CKPT=$EXP"
ls -la "$EXP" || { echo NO-CKPT; exit 1; }
sha256sum "$EXP"
unset MOTION_PIN_START_PROB MOTION_RESAMPLE_STATS
TE0=$(date +%s)
run(){ if [ "$3" = "1" ]; then export MOTION_PIN_START_FRAME=1; unset MOTION_PIN_START_FRAME_IDX MOTION_PIN_START_PROB; else unset MOTION_PIN_START_FRAME MOTION_PIN_START_FRAME_IDX MOTION_PIN_START_PROB; fi
  timeout 8m env OMNI_KIT_ACCEPT_EULA=YES /data/envs/halo-tracker/bin/python training/scripts/rsl_rl/play.py --task Halo-MotionTracking-Robot_Terrain-PreRL-Play --num_envs 1 --seed $2 --checkpoint "$1" --motion_data_folder data/g1_processed/robot-terrain/labsofa --video_length 400 --eval_json /data/halo/diag/play_ab/$4 > /data/halo/diag/play_ab/$4.log 2>&1 || { echo FAIL-$4; return 1; }; echo "OK-$4"; }
run "$EXP" 5 1 r15_exp_f0.json
run "$EXP" 1 0 r15_exp_p1.json
run "$EXP" 3 0 r15_exp_p3.json
run "$EXP" 2 0 r15_exp_p2.json
TE1=$(date +%s); echo "EVAL_SECONDS=$((TE1-TE0)) TOTAL_SECONDS=$((TE1-T0))"
echo R15-ARM-DONE
