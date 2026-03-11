# Godot Replay Client

Desktop/dev prototype for the 2D top-down broadcast vertical slice.

## What it does
- Loads deterministic replay JSON exported by the Python round simulator.
- Draws a toy-tabletop field and placeholder modular player pieces.
- Autoplays the round with camera beats, result banners, and debug odds.

## Expected workflow
1. Export a replay from the repo root:
   - `python examples/export_round_replay.py --defense-identity run-wall-zone --output apps/godot_client/replays/prototype_round_run_wall_zone.json`
   - `python examples/export_round_replay.py --defense-identity light-box-zone --output apps/godot_client/replays/prototype_round_light_box_zone.json`
2. Open `apps/godot_client/project.godot` in Godot 4.
3. Run the `Main` scene.

## Controls
- `Space`: toggle autoplay
- `Left / Right`: previous or next snap
- `R`: restart replay
- `Tab`: toggle debug probability panel

## Art pipeline
See [assets/sprites/README.md](/Users/imaad/OneDrive/Documents/Football_AutoChess/apps/godot_client/assets/sprites/README.md).
