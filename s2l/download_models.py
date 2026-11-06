"""Download every pretrained model S2L offers (Cellpose-SAM and the StarDist models).

Run by the installers:  python -m s2l.download_models
Models that are already on disk are skipped, so later runs finish instantly.
"""
import sys

from s2l.core.models_info import MODEL_SIZES_MB, all_pretrained, download_pretrained, is_downloaded


def main() -> int:
    failed = []
    for engine, name, label in all_pretrained():
        if is_downloaded(engine, name):
            print(f"[ok]   {name} ({label})")
            continue
        print(f"[get]  {name} ({label}, ~{MODEL_SIZES_MB.get(name, 5)} MB)…", flush=True)
        try:
            download_pretrained(engine, name)
            print(f"[ok]   {name}")
        except Exception as e:
            failed.append(name)
            print(f"[WARN] {name} could not be downloaded: {e}")
    if failed:
        print("Missing models are downloaded later, the first time they are used.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
