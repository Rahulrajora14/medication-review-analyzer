"""
Step 1 - Download the raw dataset.

Drug Review Dataset (Drugs.com), UCI ML Repository #462.
~215k patient reviews with drug name, condition, free-text review and 1-10 rating.
Ships as two files: drugsComTrain_raw.tsv (75%) and drugsComTest_raw.tsv (25%).

Tries the official UCI zip first, then a public GitHub mirror.
"""
import io
import urllib.request
import zipfile

from src import config


def _from_uci() -> None:
    print(f"  Trying UCI: {config.UCI_ZIP_URL}")
    with urllib.request.urlopen(config.UCI_ZIP_URL, timeout=120) as resp:
        zip_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for target in (config.RAW_TRAIN, config.RAW_TEST):
            name = next(n for n in zf.namelist() if n.endswith(target.name))
            target.write_bytes(zf.read(name))


def _from_mirror() -> None:
    for target in (config.RAW_TRAIN, config.RAW_TEST):
        url = config.MIRROR_BASE + target.name
        print(f"  Trying mirror: {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            target.write_bytes(resp.read())


def download_dataset(force: bool = False) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.RAW_TRAIN.exists() and config.RAW_TEST.exists() and not force:
        print("  Dataset already present - skipping download.")
        return

    for source in (_from_uci, _from_mirror):
        try:
            source()
            size = (config.RAW_TRAIN.stat().st_size + config.RAW_TEST.stat().st_size) / 1e6
            print(f"  Saved {size:.0f} MB -> {config.RAW_DIR}")
            return
        except Exception as err:  # noqa: BLE001 - try the next source
            print(f"  Failed: {err}")

    raise RuntimeError(
        "Automatic download failed. Download the dataset from "
        "https://archive.ics.uci.edu/dataset/462 and put both .tsv files in "
        f"{config.RAW_DIR}"
    )


if __name__ == "__main__":
    download_dataset()
