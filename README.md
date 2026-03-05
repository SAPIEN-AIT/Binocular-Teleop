# Binocular Vision Based Teleoperation and Imitation Learning Development

# BC (simple baseline)
python scripts/train.py

# ACT (action chunking — needs ≥10 demos for meaningful results)
python scripts/train.py --policy act

# Quick overrides
python scripts/train.py --policy bc --epochs 500 --lr 3e-4 --wandb