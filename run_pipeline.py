
import subprocess
import sys

STEPS = [
    "01_preprocess.py",
    "02_garch_model.py",
    "03_lstm_model.py",
    "04_compare_backtest.py",
]

for step in STEPS:
    print(f"\n{'=' * 60}\nRUNNING {step}\n{'=' * 60}")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"\n{step} failed — stopping pipeline. Fix the error above before continuing.")
        sys.exit(1)

print("\nPipeline complete. Check the outputs/ folder for every result file.")
