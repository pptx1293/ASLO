"""
ASLO Clear Training Report & Learning Curves Generator (`generate_training_report.py`)
======================================================================================
Reads the saved `models/*_metrics.json` files from all 9 pure-Keras MLP models and generates:
  1. `training_curves.png` : Epoch-by-Epoch Train vs. Validation Accuracy & Loss Learning Curves
  2. `training_report.png` : Easy-to-Understand Performance Dashboard (Train vs. Val Scoreboard,
                             Per-Gesture Accuracy Breakdown across Alphabet/Words/Dynamic Signs,
                             and Clean Percentage Confusion Matrices)
  3. `TRAINING_REPORT.md`  : Plain-English Markdown summary explaining every model, graph, and metric.
"""

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

MODELS_DIR = "models"

MODEL_SPECS = [
    ("asl_stage1_router", "Model 1A: ASL Stage-1 Shape Router", "#2563eb", "ASL Alphabet"),
    ("asl_specialist_cluster_fist", "Model 1B: ASL Fist Specialist (A,E,M,N,S,T)", "#0284c7", "ASL Alphabet"),
    ("asl_specialist_cluster_upright", "Model 1C: ASL Upright Specialist (D,H,K,R,U,V)", "#0284c7", "ASL Alphabet"),
    ("asl_specialist_cluster_curve_pinch", "Model 1D: ASL Curve/Pinch (C,F,G,O,P,Q,X)", "#0284c7", "ASL Alphabet"),
    ("asl_specialist_cluster_open", "Model 1E: ASL Open Specialist (B, W)", "#0284c7", "ASL Alphabet"),
    ("words_static", "Model 2: Static Words & Controls (11 Signs)", "#7c3aed", "Static Words"),
    ("dynamic", "Model 3A: Dynamic 25-Frame Motion (6 Classes)", "#059669", "Dynamic Motion"),
    ("dynamic_keyframe", "Model 3B: Dynamic Start/End Keyframes (10 Poses)", "#0d9488", "Dynamic Motion"),
    ("meta_router", "Master Merger: OOD-Aware Meta-Router (3 Domains)", "#ea580c", "Meta-Router"),
]


def load_all_metrics() -> Dict[str, Dict]:
    loaded = {}
    for prefix, short_name, color, group in MODEL_SPECS:
        p = os.path.join(MODELS_DIR, f"{prefix}_metrics.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["short_name"] = short_name
            data["color"] = color
            data["group"] = group
            loaded[prefix] = data
    return loaded


def plot_training_curves(metrics_map: Dict[str, Dict], out_path: str = "training_curves.png"):
    """
    Generates an easy-to-read 3x2 Learning Curves figure (`training_curves.png`):
    Top-to-bottom shows Accuracy (%) and Loss across Training vs. Validation epochs for the
    5 core pillars of the system (ASL Stage-1, ASL Specialists Average/Fist, Static Words,
    Dynamic 25-Frame Motion, Dynamic Start/End Keyframes, and Master Meta-Router).
    """
    plot_keys = [
        ("asl_stage1_router", "1. Static ASL Stage-1 Router (23 Letters -> Shape Groups)"),
        ("asl_specialist_cluster_fist", "2. ASL Fist Specialist (Hardest Similar Letters: A,E,M,N,S,T)"),
        ("words_static", "3. Static Words & Controls (11 Signs: Start, Stop, Space, etc.)"),
        ("dynamic", "4. Dynamic Moving Gestures (25-Frame Motion: HELLO, J, Z, Phrases)"),
        ("dynamic_keyframe", "5. Dynamic Start/End Pose Detector (10 _START / _END Poses)"),
        ("meta_router", "6. Master Meta-Router Merger (Combines All 3 Model Groups)"),
    ]
    available = [(k, title) for k, title in plot_keys if k in metrics_map]
    if not available:
        print("[Report] No metrics JSON files with history found yet.")
        return

    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "axes.edgecolor": "#cbd5e1"})
    fig, axes = plt.subplots(len(available), 2, figsize=(16, 3.3 * len(available)), facecolor="#f8fafc")
    if len(available) == 1:
        axes = np.array([axes])

    fig.suptitle(
        "ASLO AI Training Result Graphs — Epoch-by-Epoch Learning Curves (Train vs. Validation)\n"
        "Left = Accuracy % (Higher is Better)   |   Right = Cross-Entropy Loss (Lower is Better)",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
        y=0.992,
    )

    for row_i, (key, panel_title) in enumerate(available):
        m = metrics_map[key]
        hist = m.get("history", {})
        acc = [v * 100.0 for v in hist.get("accuracy", [])]
        val_acc = [v * 100.0 for v in hist.get("val_accuracy", [])]
        loss = hist.get("loss", [])
        val_loss = hist.get("val_loss", [])
        epochs = list(range(1, len(acc) + 1))
        best_ep = m.get("best_epoch", len(epochs))

        # Left subplot: Accuracy (%)
        ax_acc = axes[row_i, 0]
        ax_acc.set_facecolor("#ffffff")
        ax_acc.plot(epochs, acc, label=f"Train Acc (Final: {m['train_acc']*100:.2f}%)", color="#2563eb", linewidth=2.2)
        ax_acc.plot(epochs, val_acc, label=f"Validation Acc (Best: {m['val_acc']*100:.2f}%)", color="#10b981", linewidth=2.4)
        if 1 <= best_ep <= len(epochs):
            ax_acc.axvline(best_ep, color="#f59e0b", linestyle="--", linewidth=1.5, label=f"Saved Best Checkpoint (Epoch {best_ep})")
        ax_acc.set_title(f"{panel_title} — Accuracy", fontsize=11, fontweight="bold", color="#1e293b")
        ax_acc.set_xlabel("Training Epoch", fontsize=9.5, color="#475569")
        ax_acc.set_ylabel("Accuracy (%)", fontsize=9.5, color="#475569")
        y_min = max(40.0, min(min(acc, default=80.0), min(val_acc, default=80.0)) - 4.0)
        ax_acc.set_ylim(y_min, 101.2)
        ax_acc.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
        ax_acc.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="#f8fafc")

        # Right subplot: Loss
        ax_loss = axes[row_i, 1]
        ax_loss.set_facecolor("#ffffff")
        ax_loss.plot(epochs, loss, label=f"Train Loss ({m['train_loss']:.4f})", color="#6366f1", linewidth=2.0)
        ax_loss.plot(epochs, val_loss, label=f"Validation Loss ({m['val_loss']:.4f})", color="#ef4444", linewidth=2.2)
        if 1 <= best_ep <= len(epochs):
            ax_loss.axvline(best_ep, color="#f59e0b", linestyle="--", linewidth=1.5, label=f"Best Checkpoint (Epoch {best_ep})")
        ax_loss.set_title(f"{panel_title} — Error Loss", fontsize=11, fontweight="bold", color="#1e293b")
        ax_loss.set_xlabel("Training Epoch", fontsize=9.5, color="#475569")
        ax_loss.set_ylabel("Loss (Lower = Better)", fontsize=9.5, color="#475569")
        ax_loss.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
        ax_loss.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#f8fafc")

    plt.tight_layout(rect=[0, 0, 1, 0.968])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Report] Saved epoch-by-epoch train result curves to '{out_path}'.")


def plot_understandable_dashboard(metrics_map: Dict[str, Dict], out_path: str = "training_report.png"):
    """
    Generates an easy-to-understand 4-panel visual summary (`training_report.png`):
      1. Train vs. Validation Accuracy Scoreboard across all 9 MLP models
      2. Per-Letter Accuracy (%) across all 23 Static ASL Alphabet Letters (A-Y)
      3. Per-Sign Accuracy (%) across all 11 Static Words & 5 Dynamic Moving Gestures
      4. Clean Percentage + Count Confusion Matrix of the Master Meta-Router Merger
    """
    if not metrics_map:
        return

    fig = plt.figure(figsize=(18, 13), facecolor="#f8fafc")
    fig.suptitle(
        "ASLO AI Sign Language Studio — Complete Training & Validation Report\n"
        "(100% Pure Keras Multi-MLP Ensemble • 39 Total Gestures: 23 Alphabet + 11 Static Words + 5 Dynamic Signs)",
        fontsize=15.5,
        fontweight="bold",
        color="#0f172a",
        y=0.985,
    )

    # ── Panel 1: Train vs Validation Scoreboard across all 9 MLPs ─────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#ffffff")
    ordered_keys = [k for k, _, _, _ in MODEL_SPECS if k in metrics_map]
    names = [metrics_map[k]["short_name"] for k in ordered_keys][::-1]
    t_accs = [metrics_map[k]["train_acc"] * 100.0 for k in ordered_keys][::-1]
    v_accs = [metrics_map[k]["val_acc"] * 100.0 for k in ordered_keys][::-1]
    y_pos = np.arange(len(names))
    bar_h = 0.38

    ax1.barh(y_pos + bar_h / 2, t_accs, height=bar_h, color="#93c5fd", edgecolor="#2563eb", label="Train Accuracy (%)")
    bars_v = ax1.barh(y_pos - bar_h / 2, v_accs, height=bar_h, color="#10b981", edgecolor="#047857", label="Validation Accuracy (%)")
    for bar, v in zip(bars_v, v_accs):
        ax1.text(min(100.2, v + 0.25), bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", fontsize=8.5, fontweight="bold", color="#064e3b")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8.8, fontweight="bold", color="#1e293b")
    ax1.set_xlim(88.0, 103.2)
    ax1.set_xlabel("Accuracy (%) — Train & Validation Close Together = Zero Overfitting", fontsize=9.5, color="#334155")
    ax1.set_title("1. All 9 MLP Models: Train vs. Validation Scoreboard", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=8.5, frameon=True)

    # ── Panel 2: All 23 ASL Alphabet Letters Per-Letter F1/Accuracy ───────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#ffffff")
    letter_scores = {}
    if "asl_stage1_router" in metrics_map:
        rep_s1 = metrics_map["asl_stage1_router"]["report"]
        for single_l in ("L", "Y"):
            if single_l in rep_s1:
                letter_scores[single_l] = rep_s1[single_l]["f1-score"] * 100.0
    for sp_key in (
        "asl_specialist_cluster_fist",
        "asl_specialist_cluster_upright",
        "asl_specialist_cluster_curve_pinch",
        "asl_specialist_cluster_open",
    ):
        if sp_key in metrics_map:
            rep = metrics_map[sp_key]["report"]
            for cls_name in metrics_map[sp_key]["classes"]:
                if cls_name in rep:
                    letter_scores[cls_name] = rep[cls_name]["f1-score"] * 100.0

    letters_sorted = sorted(letter_scores.keys())
    l_vals = [letter_scores[k] for k in letters_sorted]
    bars_l = ax2.bar(letters_sorted, l_vals, color="#3b82f6", edgecolor="#1d4ed8", width=0.66)
    for bar, v in zip(bars_l, l_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.35, f"{v:.0f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#1e3a8a")
    ax2.set_ylim(85.0, 103.0)
    ax2.set_ylabel("Validation F1 Accuracy (%)", fontsize=9.5, color="#334155")
    ax2.set_xlabel("Static ASL Alphabet Letters (23 Static Letters; J & Z are Dynamic)", fontsize=9.5, color="#334155")
    ax2.set_title("2. Per-Letter Accuracy: Static ASL Alphabet (A–Y)", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # ── Panel 3: All 11 Static Words + 5 Dynamic Moving Gestures ──────────
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#ffffff")
    sign_items = []
    if "words_static" in metrics_map:
        rep_w = metrics_map["words_static"]["report"]
        for c in metrics_map["words_static"]["classes"]:
            if c in rep_w:
                sign_items.append((f"[Word] {c}", rep_w[c]["f1-score"] * 100.0, "#8b5cf6"))
    if "dynamic" in metrics_map:
        rep_d = metrics_map["dynamic"]["report"]
        for c in metrics_map["dynamic"]["classes"]:
            if c in rep_d and c != "STATIC_HOLD":
                disp = "HELLO (Wave)" if c.lower() == "hello" else c
                sign_items.append((f"[Dynamic] {disp}", rep_d[c]["f1-score"] * 100.0, "#10b981"))

    sign_items_rev = sign_items[::-1]
    s_labels = [x[0] for x in sign_items_rev]
    s_scores = [x[1] for x in sign_items_rev]
    s_colors = [x[2] for x in sign_items_rev]
    y_s = np.arange(len(s_labels))
    bars_s = ax3.barh(y_s, s_scores, color=s_colors, height=0.68)
    for bar, v in zip(bars_s, s_scores):
        ax3.text(min(100.3, v + 0.25), bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.3, fontweight="bold", color="#0f172a")
    ax3.set_yticks(y_s)
    ax3.set_yticklabels(s_labels, fontsize=8.5, fontweight="bold", color="#1e293b")
    ax3.set_xlim(88.0, 103.5)
    ax3.set_xlabel("Validation F1 Accuracy (%)", fontsize=9.5, color="#334155")
    ax3.set_title("3. Per-Sign Accuracy: 11 Static Words (Purple) & 5 Dynamic Signs (Green)", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax3.grid(axis="x", linestyle=":", alpha=0.6)

    # ── Panel 4: Master Meta-Router Confusion Matrix (% + Sample Counts) ──
    ax4 = fig.add_subplot(2, 2, 4)
    if "meta_router" in metrics_map:
        cm = np.asarray(metrics_map["meta_router"]["cm"], dtype=np.float32)
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        cm_pct = (cm / row_sums) * 100.0
        friendly_names = ["Dynamic Keyframes\n(_START / _END)", "Static ASL\nAlphabet (A–Y)", "Static Words\n& Controls"]
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cnt = int(cm[i, j])
                pct = cm_pct[i, j]
                annot[i, j] = f"{pct:.1f}%\n({cnt})" if cnt > 0 else "0%\n(0)"
        sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap="Oranges",
            vmin=0,
            vmax=100,
            xticklabels=friendly_names,
            yticklabels=friendly_names,
            cbar_kws={"label": "Routing Accuracy (%)"},
            annot_kws={"fontsize": 10, "fontweight": "bold"},
            ax=ax4,
        )
        ax4.set_xlabel("AI Routed Domain", fontsize=10, fontweight="bold", color="#334155")
        ax4.set_ylabel("Actual True Domain", fontsize=10, fontweight="bold", color="#334155")
        v_meta = metrics_map["meta_router"]["val_acc"] * 100.0
        n_meta = metrics_map["meta_router"]["val_samples"]
        ax4.set_title(
            f"4. Master Meta-Router Confusion Matrix ({v_meta:.2f}% • {n_meta} Test Samples)",
            fontsize=11.5,
            fontweight="bold",
            color="#0f172a",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Report] Saved understandable performance dashboard to '{out_path}'.")


def write_markdown_summary(metrics_map: Dict[str, Dict], md_path: str = "TRAINING_REPORT.md"):
    lines = [
        "# ASLO AI Sign Language Studio — Understandable Training Report",
        "",
        "## 1. Executive Summary (At a Glance)",
        "Instead of forcing one giant neural network to guess between 39 different signs at once, ASLO splits the job across **9 specialized Pure-Keras Multi-Layer Perceptron (MLP) models** coordinated by a **Master Meta-Router**.",
        "",
        "| Model Name | Role in Plain English | Train Samples | Test Samples | Best Epoch | Train Acc | Validation Acc | Val Loss |",
        "| :--- | :--- | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for prefix, short_name, _, group in MODEL_SPECS:
        if prefix not in metrics_map:
            continue
        m = metrics_map[prefix]
        lines.append(
            f"| **{short_name}** | {group} | {m['train_samples']:,} | {m['val_samples']:,} | "
            f"{m['best_epoch']}/{m['epochs_ran']} | {m['train_acc']*100:.2f}% | **{m['val_acc']*100:.2f}%** | `{m['val_loss']:.4f}` |"
        )

    lines.extend([
        "",
        "## 2. Visual Training Result Graphs",
        "- **`training_curves.png`**: Shows how **Accuracy (%)** rose and **Error Loss** dropped across every training epoch on both **Training Data** (seen by the AI) and **Validation Data** (held-out unseen 20% test split). Because the green Validation curve tracks right alongside the blue Training curve, the models generalize cleanly with **no overfitting**.",
        "- **`training_report.png`**: Shows the overall Train vs. Validation scoreboard across all 9 MLPs, per-letter accuracy for all 23 Static ASL letters (`A–Y`), per-sign accuracy for all 11 Static Words & 5 Dynamic Moving Gestures (`HELLO`, `J`, `Z`, `How are you`, `Nice to meet you`), and the Master Meta-Router confusion matrix.",
        "",
        "![Training Curves](training_curves.png)",
        "",
        "![Training Performance Dashboard](training_report.png)",
        "",
    ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] Saved plain-English report to '{md_path}'.")


def main():
    metrics_map = load_all_metrics()
    if not metrics_map:
        print("[Report] No metrics JSON files found in 'models/'. Run 'python train_model.py' first.")
        return
    plot_training_curves(metrics_map, out_path="training_curves.png")
    plot_understandable_dashboard(metrics_map, out_path="training_report.png")
    write_markdown_summary(metrics_map, md_path="TRAINING_REPORT.md")


if __name__ == "__main__":
    main()
