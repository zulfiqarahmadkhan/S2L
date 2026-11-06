"""Run one training job in its own process (used by the app's Training page).

    python -m s2l.core.train_job <job.json>

Reports on stdout, one line each:  ``PROGRESS <epoch> <total> <loss> <pct>``,
``STATUS <text>``, and at the end ``RESULT <json>`` (model path + report) or
``STOPPED``. Creating ``<job.json>.stop`` asks training to stop.

Running training outside the GUI process keeps native GPU / library problems
from ever taking the main window down, and frees all GPU memory afterwards.
"""
import json
import sys
from pathlib import Path


def main(job_file: str) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    stop_file = Path(job_file + ".stop")
    should_stop = stop_file.exists
    from s2l.core.device import set_prefer_cpu
    set_prefer_cpu(bool(job.get("prefer_cpu")))
    kw = job["kwargs"]

    def progress(epoch, total, loss, pct):
        print(f"PROGRESS {int(epoch)} {int(total)} {float(loss or 0.0):.6g} {int(pct)}", flush=True)

    def status(text):
        print(f"STATUS {text}", flush=True)

    if job["engine"] == "cellpose":
        from s2l.core.trainer import Trainer, TrainingAborted
        trainer = Trainer(train_dir=kw["train_dir"], mask_filter=kw["mask_filter"])
        try:
            path, _, report = trainer.train(
                model_name=kw["model_name"], base_model=kw["base_model"], epochs=kw["epochs"],
                learning_rate=kw["lr"], weight_decay=kw["weight_decay"], normalize=kw["normalize"],
                save_path=kw["save_path"], progress_callback=progress, should_stop=should_stop,
                test_percent=kw["test_percent"], status_callback=status)
        except TrainingAborted:
            print("STOPPED", flush=True)
            return 2
    else:
        from s2l.core.stardist_trainer import StarDistTrainer
        trainer = StarDistTrainer(train_dir=kw["train_dir"], mask_suffix=kw["mask_filter"])
        path, report = trainer.train(
            model_name=kw["model_name"], save_path=kw["save_path"], epochs=kw["epochs"],
            steps_per_epoch=kw["steps"], patch_size=tuple(kw["patch_size"]), n_rays=kw["n_rays"],
            start_from=tuple(kw["start_from"]), test_percent=kw["test_percent"],
            progress_callback=progress, should_stop=should_stop, status_callback=status)
        if should_stop():          # StarDist keeps the weights trained so far
            print("STOPPED", flush=True)
            return 2
    report["n_images"] = report["n_train"] + report["n_test"]
    print("RESULT " + json.dumps({"path": str(path), "report": report}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
