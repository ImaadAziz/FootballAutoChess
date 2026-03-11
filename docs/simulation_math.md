# Simulation Math Reference

## Overview
- The prototype round CLI uses the same core simulation math as the drive simulator.
- Every play is resolved in two layers:
  - matchup math that builds probabilities from ratings, tendencies, and context
  - outcome resolution that rolls against those probabilities in a fixed order
- The CLI now shows:
  - summary math by default
  - full ingredient scores, probability chains, and selection weights with `--show-full-math`

## Core probability helpers

### `_contest_probability(offense_score, defense_score, rng)`
- Used for matchup contests like protection vs pressure, separation vs coverage, and run block vs run defense.
- Formula:

```text
noise = Gaussian(mean=0.0, std=6.0)
contest = ((offense_score - defense_score) + noise) / 12.0
probability = clamp(sigmoid(contest), 0.01, 0.99)
```

- Higher offensive scores push the probability up.
- Higher defensive scores push the probability down.
- Random Gaussian noise keeps similar matchups from feeling deterministic.

### `_apply_logit_bias(probability, bias)`
- Used to push an event up or down in log-odds space.

```text
biased_probability = sigmoid(logit(probability) + bias)
```

### `_apply_rating_delta(baseline_probability, rating_advantage, scale, max_logit_shift)`
- Converts rating advantage into a bounded shift in log-odds space.

```text
shift = clamp(rating_advantage / scale, -max_logit_shift, max_logit_shift)
final_probability = sigmoid(logit(baseline_probability) + shift)
```

### `_resolve_probability(...)`
- Shared helper for completion, sack, interception, rush success, explosive plays, and fumbles.
- Flow:

```text
rating_probability -> model_probability -> baseline_probability -> final_probability
```

- `rating_probability` is the hand-built fallback from ratings and context.
- `model_probability` comes from the optional model bundle.
- `baseline_probability` blends rating and model using the configured model weight.
- `final_probability` applies event bias plus a rating-based logit shift.

## Pass play math

### 1. Pressure matchup
- Protection score:

```text
protection_offense =
  0.65 * avg(OL.pass_block)
  + 0.35 * QB.pocket_awareness
  + max_protect_bonus
```

- Pressure score:

```text
pressure_defense =
  0.70 * avg(rushers.pass_rush_power)
  + 0.30 * avg(rushers.get_off)
  + extra_rushers_bonus
```

- Pressure probability:

```text
pressure_prob = 1.0 - contest_probability(protection_offense, pressure_defense)
```

### 2. Separation matchup
- Receiver separation score:

```text
separation_offense =
  0.55 * avg(receivers.route_running)
  + 0.30 * avg(receivers.release)
  + 0.15 * avg(receivers.speed)
```

- Coverage score:

```text
coverage_defense =
  0.45 * avg(coverage.man_coverage)
  + 0.40 * avg(coverage.zone_coverage)
  + 0.15 * avg(coverage.play_recognition)
  + coverage-shell bonus
  + depth-specific bonus
```

- Route concept adjustment:
  - man-beaters like `drag`, `go`, `post`, `slant` help more against man looks
  - zone-beaters like `curl`, `dig`, `flat`, `out`, `sail`, `stick` help more against zone looks
  - the adjustment either raises `separation_offense` or raises `coverage_defense`
- Repetition penalty:
  - repeated exact plays, repeated depth buckets, repeated pass locations, and repeated pass-heavy sequencing add an anti-spam penalty to `coverage_defense`
  - repeated pass looks also add a smaller pressure bonus for the defense

- Separation probability:

```text
separation_prob = contest_probability(separation_offense, coverage_defense)
```

### 3. QB accuracy and decision
- QB accuracy score:

```text
qb_accuracy =
  0.45 * QB.accuracy_short
  + 0.35 * QB.accuracy_mid
  + 0.20 * QB.accuracy_deep
  - depth_penalty
```

- QB decision score:

```text
qb_decision = QB.decision - pressure_decision_penalty
```

### 4. Event probabilities
- Completion rating probability:

```text
completion_input = qb_accuracy + 0.25 * qb_decision
completion_rating_prob = contest_probability(completion_input, coverage_defense)
completion_rating_prob *= 0.63 + (0.37 * separation_prob)
completion_rating_prob = clamp(completion_rating_prob, 0.05, 0.92)
```

- Sack rating probability:

```text
sack_rating_prob = clamp(
  pressure_prob * (0.14 + 0.03 * max(0, rushers - 4)),
  0.01,
  0.38
)
```

- Interception rating probability:

```text
interception_rating_prob = (1 - completion_rating_prob) * 0.045
interception_rating_prob += pressure_bonus
interception_rating_prob += deep_shot_bonus
interception_rating_prob -= separation_bonus
interception_rating_prob = clamp(interception_rating_prob, 0.005, 0.16)
```

- Explosive catch rating probability, only after a completion:

```text
explosive_rating_prob = clamp(0.05 + ((air_yards - 10.0) / 90.0), 0.015, 0.35)
```

### 5. Resolution order
- Pass outcomes are rolled in this order:
  - under pressure?
  - sack?
  - interception?
  - completion?
  - explosive bonus after catch?
- If the play completes:

```text
air_yards = target_depth - pressure_penalty + gaussian_noise
yac = receiver_yac_model - defense_tackling_model + gaussian_noise
yards = air_yards + yac + explosive_bonus
```

### 6. Snap-level pass odds
- The CLI summary now shows true snap-level odds, not conditional event odds.

```text
snap_sack_prob = sack_prob
snap_interception_prob = (1 - sack_prob) * interception_prob
snap_completion_prob = (1 - sack_prob) * (1 - interception_prob) * completion_prob
snap_incompletion_prob = (1 - sack_prob) * (1 - interception_prob) * (1 - completion_prob)
snap_explosive_prob = snap_completion_prob * explosive_prob
```

## Run play math

### 1. Run lane matchup
- Run block score:

```text
run_block =
  avg(OL.run_block, fallback=pass_block)
  + 0.08 * RB.vision
```

- Run defense score:

```text
run_defense =
  0.55 * avg(front.run_defense, fallback=pass_rush_power)
  + 0.25 * avg(front.run_fit, fallback=play_recognition)
  + 0.20 * avg(second_level.tackling)
  + rusher_count_bonus
  + front_adjustment
```

- `front_adjustment` comes from the called front against the run direction.
  - example: `bear` fronts punish interior runs and soften some edge runs
- Repetition penalty:
  - repeated exact runs, repeated run lanes, repeated gaps, and repeated run-heavy sequencing add an anti-spam penalty to `run_defense`

- Line win probability:

```text
line_win_prob = contest_probability(run_block, run_defense)
```

### 2. Ball-carrier matchup
- RB skill score:

```text
rb_skill =
  0.40 * RB.vision
  + 0.35 * RB.elusiveness
  + 0.25 * RB.speed
```

- Pursuit+tackling composite:

```text
pursuit_tackling_score = 0.55 * pursuit + 0.45 * tackling
```

- Evade probability:

```text
evade_prob = contest_probability(rb_skill, pursuit_tackling_score)
```

### 3. Event probabilities
- Rush success rating probability starts from `line_win_prob`.
- Explosive run rating probability:

```text
explosive_rating_prob = clamp(
  (RB.speed - pursuit) / 140.0 + 0.07 * line_win_prob,
  0.008,
  0.18
)
```

- Fumble rating probability:

```text
fumble_rating_prob = clamp(
  0.004 + max(0, tackling - RB.ball_security) / 900.0,
  0.002,
  0.03
)
```

### 4. Resolution order
- Run outcomes are rolled in this order:
  - rush success?
  - evade bonus?
  - explosive bonus?
  - fumble?
- Yard construction:

```text
base_yards = success_curve if rush_success else failure_curve
yards = max(-4, base_yards + evade_bonus + explosive_bonus)
```

### 5. Snap-level run odds
- The CLI summary shows:

```text
snap_rush_success_prob = rush_success_prob
snap_evade_prob = evade_prob
snap_explosive_prob = rush_success_prob * explosive_prob
snap_fumble_prob = fumble_prob
```

## Round defense-selection math
- The round layer selects the defensive call before the play resolves.
- Inputs:
  - defense front identity like `run_wall`, `pass_rush`, `light_box`
  - defense coverage identity like `MAN`, `ZONE`, `MIXED`
  - observed offense history from the last few snaps
  - current down-and-distance
  - offense archetype tendencies as priors when there is little or no history

### Anti-spam adaptation

```text
same_play_repeats = count(last_3_snaps where play.id matches)
same_type_repeats = count(last_3_snaps where play.type matches)
adaptation_multiplier =
  1.0
  + adaptation_rate
    * (
        same_play_repeats * 0.75
        + same_type_repeats * 0.45
        + run_rate_pressure
        + deep_rate_pressure
        + short_rate_pressure
      )
```

- Repeated runs make heavy fronts more attractive.
- Repeated deep-pass tendencies make zone shells more attractive.
- Repeated short efficient passing makes matching coverage shells more attractive.

### Weighted call table
- Each defensive play starts at weight `1.0`.
- The weight is multiplied by:
  - front identity bonuses
  - coverage identity bonuses
  - observed offense tendency bonuses
  - distance situation bonuses
  - anti-spam adaptation
- The CLI full-math mode prints the final weighted table and the selection roll used to pick the call.

## CLI output mapping

### Default output
- Pass plays:
  - pressure probability
  - separation probability
  - snap-level sack, interception, completion, and explosive probabilities
- Run plays:
  - line win probability
  - snap-level rush success probability
  - snap-level evade, explosive, and fumble probabilities

### `--show-full-math`
- Adds:
  - matchup ingredient scores
  - rating -> model -> baseline -> final probability chain
  - actual resolution rolls
  - yard-build components like air yards, YAC, base yards, evade bonus, explosive bonus
  - defense weight table and anti-spam adaptation values

## No-model-bundle note
- In the prototype CLI, no external model bundle is loaded by default.
- In that case:
  - `model_probability` usually matches the hand-built `rating_probability`
  - `baseline_probability` often matches those too
- The final probability can still differ because rating shifts and configured event biases are applied after the blend step.
