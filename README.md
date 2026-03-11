# Football AutoChess

Simulation core for a football auto-battler with deterministic seeds, text-mode drive simulation, and an offline calibration/ML pipeline.

## Core features
- Situation-aware play calling (pass + run)
- Drive simulation (`simulate_drive`)
- Batch simulation (`simulate_many_drives`)
- Optional model inference via `ModelBundle`
- Calibration utilities to tune tendency knobs against target metrics

## Run the simulator
```powershell
python examples/run_single_down.py
python examples/run_text_drive.py
python examples/run_many_drives.py --drives 1000 --seed 12345
```

## Export a 2D replay
```powershell
python examples/export_round_replay.py --defense-identity run-wall-zone --output apps/godot_client/replays/prototype_round_run_wall_zone.json
python examples/export_round_replay.py --defense-identity light-box-zone --output apps/godot_client/replays/prototype_round_light_box_zone.json
```

Open `apps/godot_client/project.godot` in Godot 4 to play the exported round in the top-down replay client.

## Load real PBP with nflreadpy
```powershell
pip install nflreadpy
python examples/load_pbp_with_nflreadpy.py --seasons 2024 2025 --format parquet
```

## Data pipeline (steps 1-6)
1. Build target metrics from real play-by-play data.
2. Train event probability models (completion, sack, interception, rush success, explosive, fumble) and a pass/run play-call model.
3. Use contextual feature inputs from down/distance/field-position/clock/etc.
4. Run simulator with trained model inference.
5. Calibrate play-calling tendencies against target metrics.
6. Validate simulation distributions vs targets.

### Commands
```powershell
# Optional: fetch nflverse PBP directly with nflreadpy
python examples/load_pbp_with_nflreadpy.py --seasons 2024 2025 --output data/pbp_2024_2025.parquet

# Step 1: create target metrics from real PBP
python examples/build_targets_from_pbp.py data/pbp_2024_2025.parquet --output data/target_metrics.json

# Step 2-3: train ML models from real PBP
python examples/train_models_from_pbp.py data/pbp_2024_2025.parquet --output artifacts/model_bundle.json

# Step 4: run simulation with learned model probabilities
python examples/run_many_drives.py --drives 1000 --model-bundle artifacts/model_bundle.json

# Step 5: calibrate tendency knobs to target metrics
python examples/calibrate_simulation.py --targets data/target_metrics.json --model-bundle artifacts/model_bundle.json --output artifacts/calibration_result.json

# Step 6: validate final simulated distributions
python examples/validate_simulation.py --targets data/target_metrics.json --model-bundle artifacts/model_bundle.json --drives 2000
```

## Data format notes
- The training and target scripts accept local nflverse PBP files in `.csv`, `.csv.gz`, or `.parquet` format.
- Use `data/target_metrics.template.json` as a fallback template if you are not ready to ingest real data yet.

## New modules
- `src/football_autochess/ml_models.py`
- `src/football_autochess/pbp_pipeline.py`
- `src/football_autochess/calibration.py`
- `src/football_autochess/validation.py`
