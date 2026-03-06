# Real Play-By-Play Data Sources

Use these to bootstrap accurate training and calibration data.

## Primary source (recommended)
- nflverse data releases: https://github.com/nflverse/nflverse-data/releases
- nflverse data repo README: https://github.com/nflverse/nflverse-data
- Data dictionary reference: https://www.nflfastr.com/reference/
- nflverse docs hub: https://nflverse.nflverse.com/

Why this is the default:
- Public and widely used by football analytics projects.
- Includes full play-by-play and related tables.
- Good column coverage for simulation targets (sack, completion, INT, run/pass, explosives, etc.).

## Optional enrichment
- NFL Big Data Bowl (tracking): https://www.kaggle.com/competitions/nfl-big-data-bowl-2025

Use tracking later for:
- Route-separation realism
- Coverage leverage and pursuit modeling
- Run-fit geometry and tackle angles

## Recommended ingestion workflow
1. Easiest path: use `nflreadpy` to fetch PBP directly into `data/`.
2. Run `examples/build_targets_from_pbp.py` to compute league targets.
3. Run `examples/train_models_from_pbp.py` to train event and play-call models.
4. Run calibration and validation scripts.

## Direct loader option
```powershell
pip install nflreadpy
python examples/load_pbp_with_nflreadpy.py --seasons 2024 2025 --output data/pbp_2024_2025.parquet
```

The loader writes `parquet` by default because `nflreadpy` returns a Polars DataFrame and the local training pipeline now accepts `.parquet`, `.csv`, and `.csv.gz`.
