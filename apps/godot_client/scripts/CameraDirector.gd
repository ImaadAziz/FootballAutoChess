class_name ReplayCameraDirector
extends Node2D

@onready var camera: Camera2D = $Camera2D

var field_scene: FieldSceneController = null


func setup(field_ref: FieldSceneController) -> void:
	field_scene = field_ref
	reset_to_midfield()


func reset_to_midfield() -> void:
	if field_scene == null:
		return
	camera.global_position = field_scene.field_to_screen({"x": 50.0, "y": 26.65})
	camera.zoom = Vector2.ONE * 1.0


func play_tracks(camera_tracks: Array, actor_nodes: Dictionary, field_ref: FieldSceneController) -> void:
	field_scene = field_ref
	var tween: Tween = create_tween()
	var current_time: float = 0.0
	for track in camera_tracks:
		var track_dict: Dictionary = track as Dictionary
		var start: float = float(track_dict.get("start", current_time))
		if start > current_time:
			tween.tween_interval(start - current_time)
			current_time = start
		var focus_position: Vector2 = _resolve_focus_position(track_dict, actor_nodes)
		var zoom_value: float = float(track_dict.get("zoom", 1.0))
		var duration: float = float(track_dict.get("duration", 1.0))
		tween.parallel().tween_property(camera, "global_position", focus_position, duration)
		tween.parallel().tween_property(camera, "zoom", Vector2.ONE / maxf(0.2, zoom_value), duration)
		current_time = start + duration + float(track_dict.get("hold", 0.0))


func _resolve_focus_position(track: Dictionary, actor_nodes: Dictionary) -> Vector2:
	var focus_target: String = str(track.get("focus_target", "line_of_scrimmage"))
	if focus_target != "line_of_scrimmage" and actor_nodes.has(focus_target):
		var actor_node: Node2D = actor_nodes[focus_target] as Node2D
		return actor_node.global_position
	var pan_target: Dictionary = track.get("pan_target", {}) as Dictionary
	if not pan_target.is_empty() and field_scene != null:
		return field_scene.field_to_screen(pan_target)
	if field_scene != null:
		return field_scene.field_to_screen({"x": 50.0, "y": 26.65})
	return Vector2.ZERO
