from pathlib import Path
import os
import sys

root = Path(r'd:\Materials\SEM7\Project Work 1\Implementation\Handwriting')
ckpt = root / 'NewHandPD' / 'best_vit_finetune.pth'
hist = root / 'NewHandPD' / 'training_history.npz'
print('checkpoint exists', ckpt.exists())
print('history exists', hist.exists())
if ckpt.exists():
    print('checkpoint size', ckpt.stat().st_size)
if hist.exists():
    print('history size', hist.stat().st_size)

try:
    import subprocess
    result = subprocess.run([sys.executable, '-c', 'import psutil; import json; print(json.dumps([{"pid": p.pid, "name": p.name(), "cmdline": p.cmdline()} for p in psutil.process_iter(["name","cmdline"]) if p.name().lower().startswith("python")], indent=2))'], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print('psutil check failed', result.stderr)
except Exception as e:
    print('psutil unavailable', e)
