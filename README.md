# Warp-FWUAV

Flapping wing UAV tooling with NVIDIA Warp and XLB.

## Setup
```bash
conda env create -f env.yaml
conda activate warp-fwuav
```

## Quick Checks
```bash
pytest -q
```

## Geometry + Kinematics Preview
```bash
python scripts/preview_motion.py --spec specs/flapping_right_wing.yaml --out outputs/preview
```

Open the resulting `.vtk` files in ParaView to inspect motion.

## Static LBM Smoke Test
```bash
python scripts/run_lbm_static.py --spec specs/flapping_right_wing.yaml --n-c 40 --steps 200 --output-interval 50
```

This writes VTK field outputs to `outputs/lbm_static`.
