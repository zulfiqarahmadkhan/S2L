"""Test a segmentation model on labelled images and write publication-ready results.

Instance metrics follow the Cellpose / StarDist papers: predicted and true
objects are matched one-to-one by maximum IoU (Hungarian assignment); a match
counts as a true positive when IoU >= threshold, for thresholds 0.50…0.95.

Pixel metrics treat "inside any cell" as the positive class. AUROC / AUPRC use
the model's per-pixel cell probability (Cellpose: sigmoid of the cell-probability
output; StarDist: object probability, resized to the image).

Outputs (all in *out_dir*):
    summary_metrics.csv          one row per model × metric, with 95 % bootstrap CI
    per_image_metrics.csv        every metric for every test image
    metrics_vs_iou_threshold.csv TP / FP / FN / precision / recall / F1 / AP per IoU threshold
    roc_curve.csv, pr_curve.csv  pixel ROC and precision-recall curves
    matched_objects.csv          every true object: matched prediction and IoU
    figures/*.png (+ .svg)       ROC, PR, metrics vs IoU, count agreement, IoU histogram,
                                 per-image F1, training loss, comparison
    examples/*.png               overlays of ground truth vs prediction per test image
    README.txt                   what each file and metric means
"""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
_PIXELS_PER_IMAGE = 200_000          # pixels sampled per image for the pooled ROC / PR curves
_N_BOOT = 1000
_COLORS = ["#7c6aef", "#e5609a", "#3b9ee8", "#f0a030"]


# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════

class Predictor:
    """Wraps a Cellpose or StarDist model: ``predict(image path) -> (labels, probability map)``."""

    def __init__(self, engine: str, kind: str, value: str | None, label: str):
        self.engine, self.label = engine, label
        if engine == "cellpose":
            from cellpose import models
            from s2l.core.device import use_gpu
            self.model = models.CellposeModel(gpu=use_gpu(), pretrained_model=value or "cpsam")
        else:
            from s2l.core.stardist_segmenter import StarDistSegmenter
            self.seg = (StarDistSegmenter.from_model_dir(value) if kind == "custom"
                        else StarDistSegmenter(model_name=value))

    def predict(self, path: str):
        if self.engine == "cellpose":
            from cellpose import io
            img = io.imread(path)
            masks, flows, _ = self.model.eval(img)
            prob = 1.0 / (1.0 + np.exp(-np.asarray(flows[2], dtype=np.float32)))
            return np.asarray(masks), _fit(prob, masks.shape)
        from csbdeep.utils import normalize
        img = self.seg._read_image(path)
        ready, axes = self.seg._prepare_for_model(img)
        norm = normalize(ready, 1, 99.8, axis=(0, 1))
        (labels, _), (prob, _) = self.seg.model.predict_instances(norm, axes=axes, return_predict=True)
        return labels, _fit(prob, labels.shape)


def _fit(prob, shape):
    prob = np.squeeze(prob)
    if prob.shape != tuple(shape[:2]):
        from skimage.transform import resize
        prob = resize(prob, shape[:2], order=1, preserve_range=True, anti_aliasing=False)
    return prob.astype(np.float32)


def _read_mask(path: str) -> np.ndarray:
    from skimage.io import imread
    m = imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics of one image
# ═══════════════════════════════════════════════════════════════════════════

def _assign(iou: np.ndarray, thr: float):
    """Cellpose / StarDist matching at one threshold: maximise the number of pairs with
    IoU >= thr first, and the IoU only as a tie-break. Returns [(ti, pi, iou)] with IoU >= thr."""
    from scipy.optimize import linear_sum_assignment
    if not iou.size:
        return []
    n_min = min(iou.shape)
    costs = -(iou >= thr).astype(float) - iou / (2 * n_min)
    rows, cols = linear_sum_assignment(costs)
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols) if iou[r, c] >= thr]


def match_objects(true: np.ndarray, pred: np.ndarray):
    """IoU matrix of all objects and the one-to-one matches at IoU 0.5.

    Returns (true ids, pred ids, IoU matrix, matched pairs [(ti, pi, iou)] with IoU >= 0.5)."""
    t_ids = np.unique(true)
    t_ids = t_ids[t_ids > 0]
    p_ids = np.unique(pred)
    p_ids = p_ids[p_ids > 0]
    if not len(t_ids) or not len(p_ids):
        return t_ids, p_ids, np.zeros((len(t_ids), len(p_ids))), []
    t_map = np.zeros(true.max() + 1, dtype=np.int64)
    t_map[t_ids] = np.arange(1, len(t_ids) + 1)
    p_map = np.zeros(pred.max() + 1, dtype=np.int64)
    p_map[p_ids] = np.arange(1, len(p_ids) + 1)
    t, p = t_map[true].ravel(), p_map[pred].ravel()
    nt, np_ = len(t_ids) + 1, len(p_ids) + 1
    overlap = np.bincount(t * np_ + p, minlength=nt * np_).reshape(nt, np_)
    area_t, area_p = overlap.sum(1, keepdims=True), overlap.sum(0, keepdims=True)
    inter = overlap[1:, 1:].astype(np.float64)
    iou = inter / np.maximum(area_t[1:] + area_p[:, 1:] - inter, 1)
    return t_ids, p_ids, iou, _assign(iou, 0.5)


def image_metrics(true: np.ndarray, pred: np.ndarray, prob: np.ndarray, rng) -> dict:
    t_ids, p_ids, iou, pairs = match_objects(true, pred)
    nt, npred = len(t_ids), len(p_ids)
    # IoU of every true object with its best-overlapping prediction (for the IoU histogram)
    best = iou.max(1) if iou.size else np.zeros(nt)
    m = {"n_true": nt, "n_pred": npred, "pairs": pairs, "t_ids": t_ids, "p_ids": p_ids,
         "best_all": best, "best_iou": best[best > 0].tolist()}
    for thr in IOU_THRESHOLDS:
        tp = len(pairs) if thr == 0.5 else len(_assign(iou, thr))
        m[f"tp@{thr}"], m[f"fp@{thr}"], m[f"fn@{thr}"] = tp, npred - tp, nt - tp
    m["sum_iou@0.5"] = sum(v for *_, v in pairs)
    # pixel level
    gt, pr = true > 0, pred > 0
    m["px_tp"] = int(np.sum(gt & pr))
    m["px_fp"] = int(np.sum(~gt & pr))
    m["px_fn"] = int(np.sum(gt & ~pr))
    m["px_tn"] = int(np.sum(~gt & ~pr))
    flat_gt, flat_pr = gt.ravel(), prob.ravel()
    idx = rng.choice(flat_gt.size, min(_PIXELS_PER_IMAGE, flat_gt.size), replace=False)
    m["sample_gt"], m["sample_prob"] = flat_gt[idx], flat_pr[idx]
    m["auroc"] = m["auprc"] = float("nan")
    if 0 < flat_gt.sum() < flat_gt.size:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if 0 < m["sample_gt"].sum() < m["sample_gt"].size:     # both classes in the sample
            m["auroc"] = float(roc_auc_score(m["sample_gt"], m["sample_prob"]))
            m["auprc"] = float(average_precision_score(m["sample_gt"], m["sample_prob"]))
    return m


def _ratio(a, b):
    return float(a) / b if b else float("nan")


def instance_scores(tp, fp, fn):
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {"precision": precision, "recall": recall,
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "ap": _ratio(tp, tp + fp + fn)}       # "average precision" as defined by Cellpose / StarDist


def pixel_scores(tp, fp, fn, tn):
    return {"dice": _ratio(2 * tp, 2 * tp + fp + fn), "jaccard": _ratio(tp, tp + fp + fn),
            "pixel_accuracy": _ratio(tp + tn, tp + tn + fp + fn), "sensitivity": _ratio(tp, tp + fn),
            "specificity": _ratio(tn, tn + fp), "pixel_precision": _ratio(tp, tp + fp)}


def dataset_scores(per: list[dict]) -> dict:
    """Dataset-level metrics from per-image counts (objects pooled over images)."""
    out = {}
    aps = []
    for thr in IOU_THRESHOLDS:
        tp = sum(m[f"tp@{thr}"] for m in per)
        fp = sum(m[f"fp@{thr}"] for m in per)
        fn = sum(m[f"fn@{thr}"] for m in per)
        s = instance_scores(tp, fp, fn)
        aps.append(s["ap"])
        if thr in (0.5, 0.75):
            for k, v in s.items():
                out[f"{k}@{thr}"] = v
    out["mAP@0.5:0.95"] = float(np.nanmean(aps)) if aps else float("nan")
    tp5 = sum(m["tp@0.5"] for m in per)
    fp5 = sum(m["fp@0.5"] for m in per)
    fn5 = sum(m["fn@0.5"] for m in per)
    sum_iou = sum(m["sum_iou@0.5"] for m in per)
    out["mean_matched_iou@0.5"] = _ratio(sum_iou, tp5)
    out["panoptic_quality@0.5"] = _ratio(sum_iou, tp5 + 0.5 * fp5 + 0.5 * fn5)
    out.update(pixel_scores(*(sum(m[k] for m in per) for k in ("px_tp", "px_fp", "px_fn", "px_tn"))))
    nt = np.array([m["n_true"] for m in per], float)
    npd = np.array([m["n_pred"] for m in per], float)
    out["count_mae"] = float(np.mean(np.abs(npd - nt))) if len(nt) else float("nan")
    out["count_mean_rel_error"] = float(np.mean(np.abs(npd - nt) / np.maximum(nt, 1))) if len(nt) else float("nan")
    out["count_pearson_r"] = (float(np.corrcoef(nt, npd)[0, 1])
                              if len(nt) > 2 and nt.std() > 0 and npd.std() > 0 else float("nan"))
    out["auroc_mean_per_image"] = float(np.nanmean([m["auroc"] for m in per])) if per else float("nan")
    return out


METRIC_INFO = {
    "ap@0.5": "Average precision TP/(TP+FP+FN) at IoU 0.5 (Cellpose / StarDist definition)",
    "precision@0.5": "TP/(TP+FP) at IoU 0.5",
    "recall@0.5": "TP/(TP+FN) at IoU 0.5 (sensitivity)",
    "f1@0.5": "F1 score 2TP/(2TP+FP+FN) at IoU 0.5",
    "ap@0.75": "Average precision at IoU 0.75 (stricter outline match)",
    "precision@0.75": "Precision at IoU 0.75",
    "recall@0.75": "Recall at IoU 0.75",
    "f1@0.75": "F1 score at IoU 0.75",
    "mAP@0.5:0.95": "Mean of AP over IoU thresholds 0.50, 0.55, …, 0.95",
    "mean_matched_iou@0.5": "Mean IoU of true positive matches (segmentation quality)",
    "panoptic_quality@0.5": "Panoptic quality = sum IoU(TP) / (TP + FP/2 + FN/2)",
    "dice": "Pixel Dice coefficient, cells vs background",
    "jaccard": "Pixel IoU (Jaccard), cells vs background",
    "pixel_accuracy": "Share of pixels classified correctly",
    "sensitivity": "Pixel sensitivity (recall of cell pixels)",
    "specificity": "Pixel specificity (background correctly rejected)",
    "pixel_precision": "Pixel precision",
    "auroc": "Area under the pixel ROC curve (pooled over test images)",
    "auprc": "Area under the pixel precision-recall curve (average precision, pooled)",
    "auroc_mean_per_image": "Mean of the per-image pixel AUROC",
    "count_mae": "Mean absolute error of the number of cells per image",
    "count_mean_rel_error": "Mean relative error of the cell count per image",
    "count_pearson_r": "Pearson correlation of true vs predicted cell counts per image",
}


def _bootstrap(per: list[dict], rng) -> dict:
    """95 % CI of every dataset-level metric by resampling test images (1000×)."""
    if len(per) < 3:
        return {}
    samples = {}
    for _ in range(_N_BOOT):
        pick = [per[i] for i in rng.integers(0, len(per), len(per))]
        for k, v in dataset_scores(pick).items():
            samples.setdefault(k, []).append(v)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in samples.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation run
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(engine: str, models: list[tuple[str, str, str | None]], test_pairs: list[tuple[str, str]],
             out_dir: str, train_losses=None, progress_callback=None, should_stop=None,
             status_callback=None, n_examples: int = 12) -> dict:
    """Test each model in *models* (``[(label, kind, value)]``) on *test_pairs*; write all results.

    Returns ``{label: dataset metrics}``. The first model is "the" model;
    the others (e.g. the starting model) are evaluated for comparison.
    """
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "examples").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    truths = {}
    results, per_all, curves = {}, {}, {}
    total = len(models) * len(test_pairs)
    done = 0
    t0 = time.time()
    for label, kind, value in models:
        if status_callback:
            status_callback(f"Loading {label}…")
        predictor = Predictor(engine, kind, value, label)
        per = []
        for img_path, mask_path in test_pairs:
            if should_stop and should_stop():
                raise InterruptedError("Testing stopped by user")
            if status_callback:
                status_callback(f"{label}: {Path(img_path).name}")
            true = truths.get(mask_path)
            if true is None:
                true = truths[mask_path] = _read_mask(mask_path)
            pred, prob = predictor.predict(img_path)
            m = image_metrics(true, pred, prob, rng)
            m["image"], m["mask"] = img_path, mask_path
            m["pred"] = pred if len(per) < n_examples else None
            per.append(m)
            done += 1
            if progress_callback:
                progress_callback(int(done / total * 100))
        gt = np.concatenate([m["sample_gt"] for m in per])
        pb = np.concatenate([m["sample_prob"] for m in per])
        scores = dataset_scores(per)
        if 0 < gt.sum() < gt.size:
            fpr, tpr, _ = roc_curve(gt, pb)
            prec, rec, _ = precision_recall_curve(gt, pb)
            scores["auroc"] = float(auc(fpr, tpr))
            from sklearn.metrics import average_precision_score
            scores["auprc"] = float(average_precision_score(gt, pb))
            curves[label] = (fpr, tpr, prec, rec)
        else:
            scores["auroc"] = scores["auprc"] = float("nan")
        if status_callback:
            status_callback(f"{label}: confidence intervals…")
        ci = _bootstrap(per, rng)
        results[label] = {"scores": scores, "ci": ci}
        per_all[label] = per
        del predictor

    _write_csvs(out, models, test_pairs, results, per_all, curves)
    if status_callback:
        status_callback("Drawing figures…")
    _figures(out, results, per_all, curves, train_losses)
    _examples(out, per_all[models[0][0]], truths, n_examples)
    _readme(out, engine, models, test_pairs, results, time.time() - t0)
    return {k: v["scores"] for k, v in results.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Output files
# ═══════════════════════════════════════════════════════════════════════════

def _thin(*arrays, n=2000):
    idx = np.unique(np.linspace(0, len(arrays[0]) - 1, min(n, len(arrays[0]))).astype(int))
    return [a[idx] for a in arrays]


def _write_csvs(out, models, test_pairs, results, per_all, curves):
    with open(out / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "value", "ci95_low", "ci95_high", "n_test_images", "description"])
        for label, r in results.items():
            for k, v in r["scores"].items():
                lo, hi = r["ci"].get(k, (float("nan"), float("nan")))
                w.writerow([label, k, f"{v:.6g}", f"{lo:.6g}", f"{hi:.6g}", len(test_pairs), METRIC_INFO.get(k, "")])

    with open(out / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["model", "image", "n_true", "n_pred"]
        for thr in (0.5, 0.75):
            head += [f"tp@{thr}", f"fp@{thr}", f"fn@{thr}", f"precision@{thr}", f"recall@{thr}", f"f1@{thr}", f"ap@{thr}"]
        head += ["mean_matched_iou", "dice", "jaccard", "sensitivity", "specificity", "auroc", "auprc"]
        w.writerow(head)
        for label, per in per_all.items():
            for m in per:
                row = [label, Path(m["image"]).name, m["n_true"], m["n_pred"]]
                for thr in (0.5, 0.75):
                    tp, fp, fn = m[f"tp@{thr}"], m[f"fp@{thr}"], m[f"fn@{thr}"]
                    s = instance_scores(tp, fp, fn)
                    row += [tp, fp, fn] + [f"{s[k]:.6g}" for k in ("precision", "recall", "f1", "ap")]
                    if thr == 0.5:
                        miou = _ratio(m["sum_iou@0.5"], tp)
                px = pixel_scores(m["px_tp"], m["px_fp"], m["px_fn"], m["px_tn"])
                row += [f"{miou:.6g}"] + [f"{px[k]:.6g}" for k in ("dice", "jaccard", "sensitivity", "specificity")]
                row += [f"{m['auroc']:.6g}", f"{m['auprc']:.6g}"]
                w.writerow(row)

    with open(out / "metrics_vs_iou_threshold.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "iou_threshold", "TP", "FP", "FN", "precision", "recall", "f1", "ap"])
        for label, per in per_all.items():
            for thr in IOU_THRESHOLDS:
                tp, fp, fn = (sum(m[f"{k}@{thr}"] for m in per) for k in ("tp", "fp", "fn"))
                s = instance_scores(tp, fp, fn)
                w.writerow([label, thr, tp, fp, fn] + [f"{s[k]:.6g}" for k in ("precision", "recall", "f1", "ap")])

    for name, cols, sel in (("roc_curve.csv", ["model", "false_positive_rate", "true_positive_rate"], (0, 1)),
                            ("pr_curve.csv", ["model", "recall", "precision"], (3, 2))):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for label, c in curves.items():
                a, b = _thin(c[sel[0]], c[sel[1]])
                for x, y in zip(a, b):
                    w.writerow([label, f"{x:.6g}", f"{y:.6g}"])

    with open(out / "matched_objects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "image", "true_label", "pred_label", "iou", "status@0.5"])
        for label, per in per_all.items():
            for m in per:
                matched = {ti: (pi, v) for ti, pi, v in m["pairs"]}
                for i, tid in enumerate(m["t_ids"]):
                    if i in matched:
                        pi, v = matched[i]
                        w.writerow([label, Path(m["image"]).name, int(tid), int(m["p_ids"][pi]), f"{v:.4f}", "TP"])
                    else:   # missed: report its best overlap with any prediction
                        w.writerow([label, Path(m["image"]).name, int(tid), "", f"{m['best_all'][i]:.4f}", "FN"])
                used = {pi for _, pi, _ in m["pairs"]}
                for j, pid in enumerate(m["p_ids"]):
                    if j not in used:
                        w.writerow([label, Path(m["image"]).name, "", int(pid), "", "FP"])

    with open(out / "test_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "mask"])
        w.writerows(test_pairs)


def _new_fig(w=5.0, h=4.2):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), dpi=100, facecolor="white", layout="constrained")
    ax = fig.add_subplot(111)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def _save(fig, out, name):
    for ext in ("png", "svg"):
        fig.savefig(out / "figures" / f"{name}.{ext}", dpi=300, facecolor="white", bbox_inches="tight")


def _figures(out, results, per_all, curves, train_losses):
    labels = list(results)
    # ROC
    if curves:
        fig, ax = _new_fig()
        for i, (lab, (fpr, tpr, _, _)) in enumerate(curves.items()):
            ax.plot(fpr, tpr, color=_COLORS[i % 4], lw=2, label=f"{lab} (AUC = {results[lab]['scores']['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Pixel ROC curve",
               xlim=(0, 1), ylim=(0, 1.01))
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        _save(fig, out, "roc_curve")
        fig, ax = _new_fig()
        for i, (lab, (_, _, prec, rec)) in enumerate(curves.items()):
            ax.plot(rec, prec, color=_COLORS[i % 4], lw=2, label=f"{lab} (AP = {results[lab]['scores']['auprc']:.3f})")
        ax.set(xlabel="Recall", ylabel="Precision", title="Pixel precision-recall curve", xlim=(0, 1), ylim=(0, 1.01))
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        _save(fig, out, "pr_curve")

    # metrics vs IoU threshold
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 3.6), dpi=100, facecolor="white", layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    for ax, key, title in zip(axes, ("ap", "f1", "precision"), ("Average precision", "F1 score", "Precision")):
        for i, lab in enumerate(labels):
            per = per_all[lab]
            ys = [instance_scores(*(sum(m[f"{k}@{t}"] for m in per) for k in ("tp", "fp", "fn")))[key]
                  for t in IOU_THRESHOLDS]
            ax.plot(IOU_THRESHOLDS, ys, "o-", color=_COLORS[i % 4], lw=2, ms=4, label=lab)
        ax.set(title=title, xlabel="IoU threshold", ylim=(0, 1.02))
        ax.grid(True, color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8, frameon=False)
    _save(fig, out, "metrics_vs_iou_threshold")

    # count agreement
    fig, ax = _new_fig()
    hi = 1
    for i, lab in enumerate(labels):
        nt = [m["n_true"] for m in per_all[lab]]
        npd = [m["n_pred"] for m in per_all[lab]]
        hi = max(hi, max(nt + npd, default=1))
        r = results[lab]["scores"]["count_pearson_r"]
        ax.scatter(nt, npd, s=28, color=_COLORS[i % 4], alpha=0.8, edgecolors="none",
                   label=f"{lab}" + (f" (r = {r:.3f})" if r == r else ""))
    ax.plot([0, hi * 1.05], [0, hi * 1.05], "--", color="#999999", lw=1)
    ax.set(xlabel="True cells per image", ylabel="Predicted cells per image", title="Cell count agreement",
           xlim=(0, hi * 1.05), ylim=(0, hi * 1.05))
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out, "count_agreement")

    # IoU of matched objects
    fig, ax = _new_fig()
    for i, lab in enumerate(labels):
        ious = [v for m in per_all[lab] for v in m["best_iou"]]
        if ious:
            ax.hist(ious, bins=np.linspace(0, 1, 41), color=_COLORS[i % 4], alpha=0.55, label=lab)
    ax.axvline(0.5, color="#666666", ls="--", lw=1)
    ax.set(xlabel="Best IoU of each true cell with a prediction", ylabel="True cells",
           title="Segmentation quality per cell")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out, "iou_histogram")

    # per-image F1
    fig, ax = _new_fig(max(4.0, 1.6 * len(labels) + 2), 4.2)
    data = [[instance_scores(m["tp@0.5"], m["fp@0.5"], m["fn@0.5"])["f1"] for m in per_all[lab]] for lab in labels]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(_COLORS[i % 4])
        b.set_alpha(0.45)
    jr = np.random.default_rng(1)
    for i, d in enumerate(data):
        ax.scatter(i + 1 + jr.uniform(-0.12, 0.12, len(d)), d, s=16, color="#333333", zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    long = max(len(x) for x in labels) > 14
    ax.set_xticklabels(labels, rotation=15 if long else 0, ha="right" if long else "center")
    ax.set(ylabel="F1 at IoU 0.5", title="F1 score per test image", ylim=(0, 1.02))
    _save(fig, out, "per_image_f1")

    # comparison of the headline metrics
    if len(labels) > 1:
        keys = ["ap@0.5", "f1@0.5", "precision@0.5", "recall@0.5", "mAP@0.5:0.95", "dice", "auroc"]
        fig, ax = _new_fig(8, 4.2)
        width = 0.8 / len(labels)
        x = np.arange(len(keys))
        for i, lab in enumerate(labels):
            vals = [results[lab]["scores"].get(k, np.nan) for k in keys]
            ci = [results[lab]["ci"].get(k) for k in keys]
            err = np.array([[v - c[0], c[1] - v] if c else [0, 0] for v, c in zip(vals, ci)]).T
            ax.bar(x + (i - (len(labels) - 1) / 2) * width, vals, width, color=_COLORS[i % 4], alpha=0.85,
                   label=lab, yerr=np.clip(err, 0, None), capsize=3, error_kw={"lw": 1})
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=20, ha="right")
        ax.set(ylim=(0, 1.05), ylabel="Score", title="Model comparison (95 % CI)")
        ax.legend(fontsize=8, frameon=False)
        _save(fig, out, "comparison")

    if train_losses is not None and len(train_losses):
        fig, ax = _new_fig()
        ax.plot(np.arange(1, len(train_losses) + 1), train_losses, color=_COLORS[0], lw=1.6)
        ax.set(xlabel="Epoch", ylabel="Training loss", title="Training loss")
        _save(fig, out, "training_loss")
        with open(out / "training_loss.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss"])
            w.writerows([(i + 1, f"{v:.6g}") for i, v in enumerate(train_losses)])


def _examples(out, per, truths, n):
    """Overlay per test image: green = true cells, magenta = predicted outlines; red fill = missed cells."""
    from skimage.io import imread, imsave
    from skimage.segmentation import find_boundaries
    for m in per[:n]:
        if m.get("pred") is None:
            continue
        try:
            from s2l.core.image_files import channels_last
            img = channels_last(imread(m["image"])).astype(np.float32)
            if img.ndim == 3:
                img = img[..., :3].mean(-1)
            lo, hi = np.percentile(img, (1, 99.8))
            g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
            rgb = np.stack([g, g, g], -1)
            true, pred = truths[m["mask"]], m["pred"]
            matched = {int(m["t_ids"][ti]) for ti, _, v in m["pairs"] if v >= 0.5}
            missed = np.isin(true, [t for t in m["t_ids"] if int(t) not in matched]) & (true > 0)
            rgb[missed] = 0.6 * rgb[missed] + 0.4 * np.array([1.0, 0.15, 0.15])
            rgb[find_boundaries(true, mode="inner")] = (0.1, 0.9, 0.2)
            rgb[find_boundaries(pred, mode="inner")] = (0.95, 0.2, 0.85)
            imsave(str(out / "examples" / f"{Path(m['image']).stem}_overlay.png"),
                   (rgb * 255).astype(np.uint8), check_contrast=False)
        except Exception as e:
            logger.warning(f"Example overlay failed for {m['image']}: {e}")


def _readme(out, engine, models, test_pairs, results, seconds):
    first = models[0][0]
    s = results[first]["scores"]
    ci = results[first]["ci"]

    def fmt(k):
        v = s.get(k, float("nan"))
        c = ci.get(k)
        return f"{v:.3f}" + (f" (95% CI {c[0]:.3f}–{c[1]:.3f})" if c else "")

    lines = [
        "S2L model test results",
        "=" * 60,
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}    Engine: {engine}    Test images: {len(test_pairs)}",
        "Models: " + "; ".join(f"{lab} [{kind}: {val}]" for lab, kind, val in models),
        f"Run time: {seconds:.0f} s",
        "",
        f"Headline ({first}):",
        f"  AP@0.5 {fmt('ap@0.5')}   F1@0.5 {fmt('f1@0.5')}   mAP@0.5:0.95 {fmt('mAP@0.5:0.95')}",
        f"  Pixel Dice {fmt('dice')}   Pixel AUROC {fmt('auroc')}",
        "",
        "Methods text (for a paper):",
        f"  Segmentation accuracy was assessed on {len(test_pairs)} held-out test images that were not used",
        "  for training. Predicted and ground-truth objects were matched one-to-one by intersection over",
        "  union (IoU) using the Hungarian algorithm; a match with IoU >= t counted as a true positive (TP),",
        "  unmatched predictions as false positives (FP) and unmatched ground-truth objects as false",
        "  negatives (FN). We report average precision AP = TP/(TP+FP+FN), precision, recall and F1 at",
        "  IoU 0.5 and 0.75, mAP averaged over IoU 0.50-0.95, panoptic quality, pixel-level Dice, Jaccard,",
        "  sensitivity, specificity, and the area under the pixel ROC and precision-recall curves computed",
        "  from the model's cell-probability output. 95% confidence intervals were obtained by bootstrap",
        f"  resampling of test images ({_N_BOOT} resamples). Objects were pooled over images.",
        "",
        "Files:",
        "  summary_metrics.csv          all metrics with 95% CI and a description of each",
        "  per_image_metrics.csv        metrics for every test image",
        "  metrics_vs_iou_threshold.csv TP/FP/FN/precision/recall/F1/AP for IoU 0.50…0.95",
        "  roc_curve.csv, pr_curve.csv  pixel ROC / precision-recall curve points",
        "  matched_objects.csv          every object: TP / FN / FP and its IoU",
        "  test_images.csv              the test images and masks used",
        "  figures/                     PNG (300 dpi) and SVG (vector) figures",
        "  examples/                    overlays: green = true outline, magenta = predicted outline,",
        "                               red fill = true cell missed at IoU 0.5",
    ]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(
        {lab: r["scores"] for lab, r in results.items()}, indent=2, default=float), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Separate-process entry point (used by the app's "Test model" button)
# ═══════════════════════════════════════════════════════════════════════════

def _main(job_file: str) -> int:
    """Run one evaluation described by a JSON job file; report on stdout as
    ``STATUS <text>``, ``PROGRESS <pct>`` and finally ``RESULT <json>``.

    The app runs this in its own process so that GPU / native-library problems
    during testing can never take the main window down, and GPU memory is fully
    released afterwards. Creating a file ``<job>.stop`` asks it to stop.
    """
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    stop_file = Path(job_file + ".stop")
    from s2l.core.device import set_prefer_cpu
    set_prefer_cpu(bool(job.get("prefer_cpu")))
    try:
        res = evaluate(job["engine"], [tuple(m) for m in job["models"]], [tuple(p) for p in job["pairs"]],
                       job["out_dir"], train_losses=job.get("train_losses"),
                       progress_callback=lambda p: print(f"PROGRESS {int(p)}", flush=True),
                       should_stop=stop_file.exists,
                       status_callback=lambda s: print(f"STATUS {s}", flush=True))
    except InterruptedError:
        print("STOPPED", flush=True)
        return 2
    print("RESULT " + json.dumps(res, default=float), flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1]))
