# Toy Tabletop Sprite Pipeline

This folder is the staging area for the `AI-assisted concepts -> cleaned modular game kit` workflow.

## Folder structure
- `prompts/`: reusable prompt sheets for style boards and turnaround generation
- `concepts/`: raw generated reference sheets
- `clean/`: cleaned modular sprite parts ready for import
- `exports/`: engine-ready atlases or packed textures

## Art direction lock
- chunky helmets
- miniature plastic-body proportions
- readable jersey contrast at broadcast scale
- strong base plates and shadows
- simplified limbs that look good with procedural motion

## MVP archetypes
- QB
- skill player
- OL / TE heavy body
- front-seven defender
- DB defender

## Quality rule
- Use AI for concept exploration and turnaround references.
- Do not use raw generated sprite sheets directly in-game.
- Clean and standardize final parts into layers: helmet, head, torso, arms, legs, cleats, decals, base/shadow.
- Recolor inside Godot so team identity comes from palette data, not duplicate textures.
