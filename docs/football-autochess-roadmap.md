# 3D American Football Auto-Chess Roadmap

## Goal
Build a football auto-battler where each round simulates one down. Offense has 4 downs to gain 10 yards, resets downs on conversion, and continues until touchdown, turnover, or end-of-drive condition.

## Design Principles
- Sim first, visuals second: build a deterministic + random simulation core before full 3D presentation.
- Skill-driven randomness: ratings should shape outcomes, but no play should be guaranteed.
- Clear phases: each down resolves in ordered steps so outcomes are explainable and tunable.
- Depth over complexity: start with pass game + pass defense loop, then layer run game and advanced tactics.

## Core Down Simulation (Round Loop)
1. Situation setup
- Inputs: down, distance, field position, clock, score, fatigue, previous play tendency.

2. Play call selection (auto-chess decision)
- Offense selects a play from its playbook based on situation and team tendency.
- Defense selects coverage/front package based on offense tendency and situation.

3. Matchup initialization
- Assign blockers to rushers.
- Assign coverage responsibilities (man or zone).
- Assign route trees and timing windows.

4. Trench battle (OL vs DL/EDGE)
- Simulate pass protection over time slices.
- Output: pocket integrity and pressure arrival time.

5. Route/coverage battle (WR/TE/RB vs DB/LB)
- Track route separation by time and coverage type.
- Zone: evaluate windows by area.
- Man: evaluate direct defender separation.

6. QB decision phase
- QB reads progression, reacts to pressure, and selects throw/checkdown/scramble/throwaway.
- Decision quality depends on awareness + processing + pressure.

7. Ball outcome phase
- Resolve accuracy, pass breakup/interception chance, catch chance, and contested catch.

8. After-catch phase
- Resolve YAC and tackle pursuit to determine final yardage.

9. Game-state update
- Apply result, update down/distance/field position/clock.
- If yards >= distance: first down and reset to 1st-and-10.
- If down > 4 without conversion: turnover on downs.
- End drive on touchdown/turnover/safety/etc.

## Rating Model (First Pass)
Use 0-100 ratings.

- QB: `accuracy_short`, `accuracy_mid`, `accuracy_deep`, `decision`, `pocket_awareness`, `arm_strength`, `mobility`
- WR/TE/RB: `release`, `route_running`, `speed`, `catch`, `contested_catch`, `yac`, `pass_block`
- OL: `pass_block`, `anchor`, `awareness`
- DL/EDGE: `pass_rush_finesse`, `pass_rush_power`, `get_off`, `pursuit`
- LB/DB: `man_coverage`, `zone_coverage`, `play_recognition`, `ball_skills`, `tackling`
- Team/Coach: `play_calling`, `discipline`, `tempo`, `aggressiveness`

## Randomness + Skill Formula
Use weighted contest scores and a bounded probability function:

```text
contest = (skill_weighted_offense - skill_weighted_defense) + situational_mods + random_noise
win_prob = 1 / (1 + exp(-contest / scale))
```

Notes:
- `random_noise` should be small-to-moderate so ratings matter long-term.
- Use larger variance on explosive outcomes (deep balls, broken tackles).
- Keep seeds configurable for deterministic replay/testing.

## Play Archetypes (MVP)
Offense:
- Quick Game (slants, hitches)
- Intermediate Concepts (dagger, crossers)
- Deep Shot
- Screen/Checkdown

Defense:
- Cover 1 Man
- Cover 2 Zone
- Cover 3 Zone
- 4-man rush / 5-man pressure package

## Phase Plan
### Phase 1: Simulation Core (No 3D)
- Build data models for players, teams, plays, and game state.
- Implement `simulate_down()` with the 9-step loop.
- Add drive loop for downs + first-down reset.
- Output detailed play-by-play logs.

### Phase 2: Tuning + AI Play Calling
- Add weighted play selection based on situation.
- Add tendency tracking and simple adaptation.
- Run bulk simulations to tune balance.

### Phase 3: 3D Integration
- Map simulation events to animations (routes, drops, pressure, throws, tackles).
- Keep gameplay authority in simulation core; 3D layer is presentation.

### Phase 4: Depth
- Add run game and run fits.
- Add fatigue, traits, chemistry, and injuries.
- Add scouting/meta progression for auto-chess depth.

## Implementation Checklist
- [x] Define core schema (`Player`, `Team`, `Play`, `GameState`, `PlayResult`)
- [x] Build offense/defense playbook format
- [x] Implement down/drive state transitions
- [x] Add deterministic seed support
- [ ] Implement matchup resolver (OL-DL, WR-DB)
- [ ] Implement QB decision engine
- [ ] Implement pass outcome resolver
- [ ] Implement YAC+tackle resolver
- [x] Add simulation test harness (1,000+ drives)
- [ ] Tune rating weights and variance
- [ ] Connect events to 3D animation layer

## Progress Log
### 2026-03-05
- Added Python package scaffold in `src/football_autochess`.
- Implemented `simulate_down(...)` with seeded randomness and per-play event logs.
- Added situation-aware offensive and defensive play selection.
- Added simple tendency tracking/adaptation using recent play history.
- Implemented `simulate_drive(...)` to run a full text-mode drive.
- Added runnable examples in `examples/run_single_down.py` and `examples/run_text_drive.py`.
- Added initial run-play resolver for inside/edge run outcomes.
- Added `simulate_many_drives(...)` benchmark harness with aggregate rate metrics.
- Added `examples/run_many_drives.py` to run 1,000-drive balance checks.
- Added offline ML pipeline (PBP ingestion, event-model training, calibration, and validation scripts).
- Added optional model-bundle inference to `simulate_down`, `simulate_drive`, and `simulate_many_drives`.

## Suggested Next Step
Implement calibration and validation tooling:

```text
calibrate_weights(batch_metrics, target_stats) -> updated_config
``` 

Then connect the calibrated simulation outputs to 3D animation states (routes, run fits, pressure, throws, tackles) with replay logging for debugging.




