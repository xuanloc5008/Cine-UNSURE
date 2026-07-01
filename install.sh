find /workspace/datasets -type f -name '._*' -delete
find /workspace/datasets -type f -name '.DS_Store' -delete

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