class_name ReplaySnapDirector
extends Node

const PLAYER_PIECE_SCENE: PackedScene = preload("res://scenes/PlayerPiece.tscn")


func play_snap(snap_payload: Dictionary, field_scene: FieldSceneController, camera_director: ReplayCameraDirector, hud: ReplayHUD) -> void:
	field_scene.set_snap_context(snap_payload)
	field_scene.clear_snap()
	var actor_nodes: Dictionary = _spawn_actors(snap_payload.get("actors", []) as Array, field_scene)
	camera_director.play_tracks(snap_payload.get("camera", []) as Array, actor_nodes, field_scene)
	_play_paths(snap_payload.get("paths", []) as Array, actor_nodes, field_scene)

	var banner_start: float = _banner_start_time(snap_payload.get("beats", []) as Array)
	var total_duration: float = _snap_duration(snap_payload)
	if banner_start > 0.0:
		await get_tree().create_timer(banner_start).timeout
	hud.set_post_snap(snap_payload)
	if total_duration > banner_start:
		await get_tree().create_timer(total_duration - banner_start).timeout


func _spawn_actors(actors_payload: Array, field_scene: FieldSceneController) -> Dictionary:
	var actor_nodes: Dictionary = {}
	for actor_payload in actors_payload:
		var actor_dict: Dictionary = actor_payload as Dictionary
		var piece: ReplayPlayerPiece = PLAYER_PIECE_SCENE.instantiate() as ReplayPlayerPiece
		field_scene.actor_layer.add_child(piece)
		piece.configure(actor_dict)
		piece.set_screen_position(field_scene.field_to_screen(actor_dict.get("start_position", {"x": 0.0, "y": 0.0}) as Dictionary))
		actor_nodes[actor_dict.get("id", "")] = piece
	return actor_nodes


func _play_paths(paths_payload: Array, actor_nodes: Dictionary, field_scene: FieldSceneController) -> void:
	for path_payload in paths_payload:
		var path_dict: Dictionary = path_payload as Dictionary
		var actor_id: String = str(path_dict.get("actor_id", ""))
		if not actor_nodes.has(actor_id):
			continue
		var node: Node2D = actor_nodes[actor_id] as Node2D
		var points: Array = path_dict.get("points", []) as Array
		if points.is_empty():
			continue
		var tween: Tween = create_tween()
		tween.tween_interval(float(path_dict.get("start", 0.0)))
		var segment_duration: float = float(path_dict.get("duration", 0.4)) / float(max(1, points.size()))
		for point_payload in points:
			tween.tween_property(node, "global_position", field_scene.field_to_screen(point_payload as Dictionary), segment_duration)


func _banner_start_time(beats_payload: Array) -> float:
	for beat_payload in beats_payload:
		var beat_dict: Dictionary = beat_payload as Dictionary
		var banner_text: String = str(beat_dict.get("banner_text", ""))
		if not banner_text.is_empty():
			return float(beat_dict.get("start", 0.0))
	return 0.0


func _snap_duration(snap_payload: Dictionary) -> float:
	var duration: float = 0.0
	for path_payload in snap_payload.get("paths", []):
		var path_dict: Dictionary = path_payload as Dictionary
		duration = maxf(duration, float(path_dict.get("start", 0.0)) + float(path_dict.get("duration", 0.0)))
	for beat_payload in snap_payload.get("beats", []):
		var beat_dict: Dictionary = beat_payload as Dictionary
		duration = maxf(duration, float(beat_dict.get("start", 0.0)) + float(beat_dict.get("duration", 0.0)))
	for camera_payload in snap_payload.get("camera", []):
		var camera_dict: Dictionary = camera_payload as Dictionary
		duration = maxf(duration, float(camera_dict.get("start", 0.0)) + float(camera_dict.get("duration", 0.0)) + float(camera_dict.get("hold", 0.0)))
	return maxf(4.2, duration)
