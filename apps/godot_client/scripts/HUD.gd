class_name ReplayHUD
extends CanvasLayer

var debug_visible: bool = false

@onready var matchup_label: Label = $Root/TopBar/MatchupLabel
@onready var status_label: Label = $Root/TopBar/StatusLabel
@onready var tell_label: Label = $Root/BottomCard/TellLabel
@onready var play_label: Label = $Root/BottomCard/PlayLabel
@onready var result_label: Label = $Root/BottomCard/ResultLabel
@onready var debug_label: Label = $Root/BottomCard/DebugLabel
@onready var controls_label: Label = $Root/BottomCard/ControlsLabel


func set_round_header(meta_payload: Dictionary, teams_payload: Dictionary) -> void:
	var offense: Dictionary = teams_payload.get("offense", {}) as Dictionary
	var defense: Dictionary = teams_payload.get("defense", {}) as Dictionary
	matchup_label.text = "%s vs %s" % [offense.get("name", "Offense"), defense.get("name", "Defense")]
	status_label.text = "Round: %s | Goal: %s yards in %s plays" % [
		meta_payload.get("round_name", "Prototype Round"),
		meta_payload.get("target_yards", 40),
		meta_payload.get("play_budget", 6),
	]
	controls_label.text = "Space autoplay | Left/Right snap | R restart | Tab debug"


func set_pre_snap(snap_payload: Dictionary) -> void:
	var pre_snap: Dictionary = snap_payload.get("pre_snap", {}) as Dictionary
	var ui_payload: Dictionary = snap_payload.get("ui", {}) as Dictionary
	var scoreboard: Dictionary = ui_payload.get("scoreboard", {}) as Dictionary
	status_label.text = "Q%s %s | %s | %s yards left | %s plays left" % [
		scoreboard.get("quarter", 1),
		_format_clock(int(scoreboard.get("clock_seconds", 900))),
		scoreboard.get("down_text", "1 & 40"),
		scoreboard.get("yards_remaining", pre_snap.get("distance", 40)),
		scoreboard.get("plays_left", pre_snap.get("plays_left", 6)),
	]
	tell_label.text = "Pre-snap tell: %s" % pre_snap.get("tell_text", "")
	play_label.text = "Call: %s" % ui_payload.get("play_banner", "")
	result_label.text = ""
	_set_debug_text(ui_payload.get("debug_panel", {}))


func set_post_snap(snap_payload: Dictionary) -> void:
	var outcome: Dictionary = snap_payload.get("outcome", {}) as Dictionary
	result_label.text = "Result: %s (%s)" % [outcome.get("summary", ""), _yards_text(int(outcome.get("yards_gained", 0)))]
	status_label.text = "Next: %s & %s | Ball @ %s" % [
		outcome.get("next_down", 1),
		outcome.get("next_distance", 0),
		outcome.get("next_yard_line", 25),
	]


func toggle_debug() -> void:
	debug_visible = not debug_visible
	debug_label.visible = debug_visible


func _set_debug_text(debug_panel: Dictionary) -> void:
	var lines: Array[String] = []
	var probabilities: Dictionary = debug_panel.get("probabilities", {}) as Dictionary
	for key in probabilities.keys():
		var key_text: String = str(key)
		lines.append("%s: %s%%" % [key_text.capitalize(), snappedf(float(probabilities[key]) * 100.0, 0.1)])
	debug_label.text = "\n".join(lines)
	debug_label.visible = debug_visible


func _format_clock(seconds: int) -> String:
	var mins: int = int(max(0, seconds) / 60)
	var secs: int = int(max(0, seconds) % 60)
	return "%02d:%02d" % [mins, secs]


func _yards_text(yards: int) -> String:
	if yards > 0:
		return "+%s" % yards
	return str(yards)
