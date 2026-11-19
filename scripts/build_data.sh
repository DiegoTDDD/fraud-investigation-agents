#!/usr/bin/env bash
# Build the transactions and the fraud model exactly as realtime-fraud-detection-ml does, at a
# pinned commit, then check cases/cases.json is reproduced from them.
#
#   bash scripts/build_data.sh
set -euo pipefail

REPO=https://github.com/DiegoTDDD/realtime-fraud-detection-ml
COMMIT=3ebb8bba1cce094070b6411a14442bcab8b16a0e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone --quiet "$REPO" "$WORK"
git -C "$WORK" checkout --quiet "$COMMIT"
(
  cd "$WORK"
  mkdir -p data docs/screenshots
  python training/generate_training_data.py
  python training/train_model.py
)
mkdir -p "$ROOT/data" "$ROOT/models/fraud"
cp "$WORK/data/training_raw.parquet" "$ROOT/data/transactions.parquet"
cp "$WORK/training/card_home_city.json" "$ROOT/data/"
cp "$WORK"/training/{model.joblib,card_home_city.json,metrics.json} "$ROOT/models/fraud/"
echo "$REPO@$COMMIT" > "$ROOT/models/fraud/SOURCE"

cp "$ROOT/cases/cases.json" "$WORK/cases.committed.json"
python "$ROOT/scripts/build_cases.py" "$ROOT/data/transactions.parquet"
diff -q "$WORK/cases.committed.json" "$ROOT/cases/cases.json" \
  || { echo "cases/cases.json changed: the generator or the pin moved"; exit 1; }
