find /workspace/datasets \
    -type f \
    \( -name '._*' -o -name '.DS_Store' \) \
    -delete
# python -m pip uninstall -y opendatasets kaggle kagglesdk
# python -m pip install "kaggle==1.6.17"

cd /workspace/Cine-UNSURE
#==============================

python - <<'PY'
from pathlib import Path

p = Path("pyproject.toml")
text = p.read_text()
lines = text.splitlines()

out = []
seen = False
skip_next_packages = False

i = 0
while i < len(lines):
    line = lines[i].strip()
    if line == "[tool.setuptools]":
        if seen:
            i += 1
            if i < len(lines) and lines[i].strip().startswith("packages"):
                i += 1
            continue
        seen = True
        out.append(lines[i])
        i += 1
        continue
    out.append(lines[i])
    i += 1

text = "\n".join(out).rstrip() + "\n"
if "[tool.setuptools]" not in text:
    text += '\n[tool.setuptools]\npackages = ["score_cunsure"]\n'
elif 'packages = ["score_cunsure"]' not in text:
    text += '\npackages = ["score_cunsure"]\n'

p.write_text(text)
PY

python -m pip install -e ".[all]"

#==============================
printf '\n[tool.setuptools]\npackages = ["score_cunsure"]\n' >> pyproject.toml
python -m pip install -e ".[all]"

# install model
python scripts/download_checkpoints.py

cd /workspace/Cine-UNSURE

mkdir -p work/external

if [ ! -d work/external/CineMA ]; then
  git clone https://github.com/mathpluscode/CineMA.git work/external/CineMA
fi

python -m pip install -e work/external/CineMA

python - <<'PY'
import sys
sys.path.insert(0, "/workspace/Cine-UNSURE/work/external/CineMA")

from cinema import CineMA

model = CineMA.from_pretrained()
model.eval()
print("CineMA loaded OK")
PY

find /work/external/MedSAM2/checkpoints -maxdepth 1 -type f | sort