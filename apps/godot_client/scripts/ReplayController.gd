class_name ReplayControllerRoot
extends Node2D

@export_file("*.json") var replay_path: String = "res://replays/prototype_round_run_wall_zone.json"
@export var autoplay: bool = true

var replay_data: Dictionary = {}
var snap_index: int = 0
var is_playing: bool = false

@onready var field_scene: FieldSceneController = $Field
@onready var snap_director: ReplaySnapDirector = $SnapDirector
@onready var camera_director: ReplayCameraDirector = $CameraDirector
@onready var hud: ReplayHUD = $HUD


func _ready() -> void:
	replay_data = _load_replay(replay_path)
	field_scene.setup(replay_data.get("field", {}) as Dictionary, replay_data.get("teams", {}) as Dictionary)
	camera_director.setup(field_scene)
	hud.set_round_header(replay_data.get("meta", {}) as Dictionary, replay_data.get("teams", {}) as Dictionary)
	_play_current_snap()


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_accept"):
		autoplay = not autoplay
	elif event.is_action_pressed("ui_right"):
		autoplay = false
		next_snap()
	elif event.is_action_pressed("ui_left"):
		autoplay = false
		previous_snap()
	elif event.is_action_pressed("ui_cancel"):
		restart()
	elif event.is_action_pressed("ui_focus_next"):
		hud.toggle_debug()


func next_snap() -> void:
	if is_playing:
		return
	var snaps: Array = replay_data.get("snaps", []) as Array
	snap_index = mini(snap_index + 1, snaps.size() - 1)
	_play_current_snap()


func previous_snap() -> void:
	if is_playing:
		return
	snap_index = maxi(snap_index - 1, 0)
	_play_current_snap()


func restart() -> void:
	if is_playing:
		return
	snap_index = 0
	_play_current_snap()


func _play_current_snap() -> void:
	if replay_data.is_empty():
		return
	var snaps: Array = replay_data.get("snaps", []) as Array
	if snap_index < 0 or snap_index >= snaps.size() or is_playing:
		return

	is_playing = true
	var snap_payload: Dictionary = snaps[snap_index] as Dictionary
	hud.set_pre_snap(snap_payload)
	await snap_director.play_snap(snap_payload, field_scene, camera_director, hud)
	is_playing = false

	if autoplay and snap_index < snaps.size() - 1:
		snap_index += 1
		call_deferred("_play_current_snap")


func _load_replay(path_text: String) -> Dictionary:
	var file: FileAccess = FileAccess.open(path_text, FileAccess.READ)
	if file == null:
		push_error("Unable to open replay file: %s" % path_text)
		return {}
	var parsed: Variant = JSON.parse_string(file.get_as_text())
	if typeof(parsed) != TYPE_DICTIONARY:
		push_error("Replay file did not parse into a dictionary: %s" % path_text)
		return {}
	return parsed as Dictionary
