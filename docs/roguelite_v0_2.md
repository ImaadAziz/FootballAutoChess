# Football Autochess Roguelite v0.2

## High concept
- Offense-only football roguelite with auto-battler execution.
- The player calls each play, but the roster executes automatically based on ratings, traits, matchups, and fatigue.
- The goal is to beat a ladder of increasingly specialized defenses.

## Core round rules
- Each round starts from a fixed distance to the end zone.
- Example: `1st and 40` means the offense needs 40 total yards to score.
- There are no first downs in this mode.
- After each snap, the offense advances to the next down number with the remaining yards to go.
- Example: `1st and 40 -> gain 20 -> 2nd and 20`.
- Rounds use a strict total-play budget.
- Example MVP target: `40 yards in 6 plays`.
- A round is won by scoring before the play budget runs out.
- A round is lost if:
  - the offense turns the ball over
  - the play budget reaches zero without a touchdown

## Run structure
- MVP uses a straight ladder instead of a map.
- A run ends after `3 failed drives`.
- Winning faster gives better rewards.
- Later rounds increase distance, tighten the play budget, or field more specialized defenses.

## Starter offense archetypes
- `Precision Pass`
  - Elite QB
  - Strong at short and intermediate passing, quick reads, and efficient execution
- `Ground Control`
  - Elite RB
  - Strong at steady rushing gains, play action, and fatigue resistance
- `Vertical Attack`
  - Elite WR
  - Strong at explosive deep plays and punishing man coverage

## Defense identity rules
- Every defense has two readable layers:
  - `Front identity`
  - `Coverage identity`
- Front identity examples:
  - `Run Wall`
  - `Pass Rush`
  - `Light Box`
  - `Balanced`
- Coverage identity examples:
  - `Man`
  - `Zone`
  - `Mixed / Disguise`
- Strong defenses must also have a visible weakness the offense can exploit.
- Example:
  - A `Run Wall + Zone` defense should pack the box and squeeze inside runs, but invite edge runs and underneath spacing.

## MVP tactical systems
- Defense adaptation should punish repeated play calls and repeated concepts.
- Fatigue should make overusing the same player or same concept less effective.
- Route concepts and player traits should matter versus man and zone.
- Audibles should be strong but limited, and tied to specific players.

## Prototype target
- First playable prototype: one text-mode round.
- Recommended setup:
  - `Precision Pass` offense
  - `Run Wall + Zone Shell` defense
  - `40 yards to score`
  - `6 total plays`
