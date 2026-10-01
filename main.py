"""
Run the whole pipeline with one command:

    python main.py                  # every step (DistilBERT only if a GPU is found)
    python main.py --step baseline  # one step
    python main.py --from eda       # from a step to the end
    python main.py --step transformer --allow-cpu   # force DistilBERT on CPU (slow!)

Steps:  download -> prepare -> eda -> baseline -> transformer -> report -> errors
"""
import argparse
import time

STEPS = ["download", "prepare", "eda", "baseline", "transformer", "report", "errors"]
TITLES = {
    "download": "Download the dataset",
    "prepare": "Clean, label, de-duplicate and split",
    "eda": "Exploratory analysis + drug insights",
    "baseline": "Train TF-IDF + Logistic Regression baseline",
    "transformer": "Fine-tune DistilBERT (PyTorch)",
    "report": "Compare models + update README",
    "errors": "Error analysis",
}


def run_step(name: str, allow_cpu: bool) -> None:
    # imports live inside each branch so e.g. the baseline works without torch installed
    if name == "download":
        from src.data.download_data import download_dataset
        download_dataset()
    elif name == "prepare":
        from src.data.prepare import prepare_data
        prepare_data()
    elif name == "eda":
        from src.analysis.eda import run_eda
        run_eda()
    elif name == "baseline":
        from src.models.baseline import train_baseline
        train_baseline()
    elif name == "transformer":
        try:
            import torch
        except Exception as err:  # not installed OR installed but broken (e.g. CUDA mismatch)
            print(f"  PyTorch could not be imported ({err.__class__.__name__}) - skipping.\n"
                  "  See README > Fine-tune DistilBERT on a free GPU.")
            return
        if not torch.cuda.is_available() and not allow_cpu:
            print("  No GPU found - skipping DistilBERT (hours on CPU).\n"
                  "  Run it free on Kaggle/Colab: notebooks/finetune_distilbert_gpu.ipynb\n"
                  "  or force it here with: python main.py --step transformer --allow-cpu")
            return
        from src.models.transformer import train_transformer
        train_transformer()
    elif name == "report":
        from src.models.report import run_report
        run_report()
    elif name == "errors":
        from src.models.error_analysis import run_error_analysis
        run_error_analysis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medication review analyzer pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", choices=STEPS, help="run only this step")
    group.add_argument("--from", dest="start", choices=STEPS, help="run from this step onward")
    parser.add_argument("--allow-cpu", action="store_true", help="allow DistilBERT on CPU")
    args = parser.parse_args()

    steps = [args.step] if args.step else STEPS[STEPS.index(args.start):] if args.start else STEPS
    t0 = time.time()
    for i, name in enumerate(steps, 1):
        print(f"\n{'=' * 64}\n[{i}/{len(steps)}] {TITLES[name]}\n{'=' * 64}")
        t = time.time()
        run_step(name, args.allow_cpu)
        print(f"  done in {time.time() - t:.1f}s")
    print(f"\nFinished in {time.time() - t0:.1f}s. Next: streamlit run app/streamlit_app.py")
