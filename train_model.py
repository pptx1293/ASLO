"""
ASLO Multi-MLP Master Training Orchestrator (`train_model.py`)
==============================================================
Runs the 3 completely separate standalone trainers and then merges all 3 model groups
using the dedicated Meta-Router MLP (`merge_models.py`):
  1. `train_asl_model.py`     -> Pure Static ASL (Stage-1 Router + 4 Stage-2 Specialists)
  2. `train_words_model.py`   -> Pure Static Words & Control Gestures (`words_static_mlp.keras`)
  3. `train_dynamic_model.py` -> Pure Dynamic Gestures (25-Frame `dynamic_mlp.keras` + `_START/_END` `dynamic_keyframe_mlp.keras`)
  4. `merge_models.py`        -> Meta-Router MLP (`meta_router_mlp.keras`) + `merged_ensemble_manifest.json`

Usage:
  python train_model.py                 # Train all 3 models separately, then merge
  python train_model.py --only asl      # Train ONLY Static ASL model group
  python train_model.py --only words    # Train ONLY Static Words model group
  python train_model.py --only dynamic  # Train ONLY Dynamic Gestures model group
  python train_model.py --only merge    # Merge existing trained models via Meta-Router MLP
"""

import argparse

import generate_training_report
import merge_models
import train_asl_model
import train_dynamic_model
import train_words_model


def main():
    parser = argparse.ArgumentParser(description="ASLO Separate-Training & Merged Multi-MLP Pipeline")
    parser.add_argument(
        "--only",
        choices=["all", "asl", "words", "dynamic", "merge"],
        default="all",
        help="Run a specific standalone trainer or the full separate-train + merge pipeline.",
    )
    args = parser.parse_args()

    asl_res = None
    words_res = None
    dyn_res = None
    merge_res = None

    if args.only in ("all", "asl"):
        asl_res = train_asl_model.main()
    if args.only in ("all", "words"):
        words_res = train_words_model.main()
    if args.only in ("all", "dynamic"):
        dyn_res = train_dynamic_model.main()
    if args.only in ("all", "merge"):
        merge_res = merge_models.main()

    if args.only == "all" and asl_res and words_res and dyn_res and merge_res:
        print("\n" + "=" * 72)
        print("SEPARATE-TRAINING + MERGED MULTI-MLP ARCHITECTURE SUMMARY")
        print("=" * 72)
        print(f"1. Static ASL Stage-1 Router MLP    : Val Acc = {asl_res['stage1']['val_acc']*100:6.2f}%")
        for c_name, c_st in asl_res["specialists"].items():
            print(f"   * Stage-2 {c_name:<22} : Val Acc = {c_st['val_acc']*100:6.2f}% ({c_st['classes']})")
        print(f"2. Pure Static Words MLP            : Val Acc = {words_res['words']['val_acc']*100:6.2f}%")
        print(f"3. Dynamic 25-Frame Trajectory MLP  : Val Acc = {dyn_res['dynamic_trajectory']['val_acc']*100:6.2f}%")
        print(f"   * Dynamic Keyframes (_START/_END): Val Acc = {dyn_res['dynamic_keyframe']['val_acc']*100:6.2f}%")
        print(f"4. Merged Meta-Router MLP           : Val Acc = {merge_res['meta_router']['val_acc']*100:6.2f}%")
        print("=" * 72 + "\n")

        generate_training_report.main()
    else:
        # Also refresh visual reports whenever any subset finishes and metrics exist
        generate_training_report.main()


if __name__ == "__main__":
    main()
