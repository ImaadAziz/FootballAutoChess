class_name FieldSceneController
extends Node2D

const FIELD_DRAW_SIZE: Vector2 = Vector2(1360.0, 720.0)
const FIELD_MARGIN: Vector2 = Vector2(120.0, 96.0)

var field_data: Dictionary = {}
var team_data: Dictionary = {}
var current_snap: Dictionary = {}

@onready var actor_layer: Node2D = $Actors
@onready var overlay_layer: Node2D = $Overlays


func setup(field_payload: Dictionary, teams_payload: Dictionary) -> void:
	field_data = field_payload.duplicate(true)
	team_data = teams_payload.duplicate(true)
	queue_redraw()


func set_snap_context(snap_payload: Dictionary) -> void:
	current_snap = snap_payload.duplicate(true)
	queue_redraw()


func clear_snap() -> void:
	for child in actor_layer.get_children():
		child.queue_free()
	for child in overlay_layer.get_children():
		child.queue_free()


func field_to_screen(point: Dictionary) -> Vector2:
	var length: float = float(field_data.get("length", 100.0))
	var width: float = float(field_data.get("width", 53.3))
	var x: float = FIELD_MARGIN.x + (float(point.get("x", 0.0)) / length) * FIELD_DRAW_SIZE.x
	var y: float = FIELD_MARGIN.y + (float(point.get("y", 0.0)) / width) * FIELD_DRAW_SIZE.y
	return Vector2(x, y)


func _draw() -> void:
	var field_rect: Rect2 = Rect2(FIELD_MARGIN, FIELD_DRAW_SIZE)
	draw_rect(field_rect, Color("0f5d31"), true)
	draw_rect(field_rect.grow(6.0), Color("082716"), false, 6.0)

	var end_zone_width: float = FIELD_DRAW_SIZE.x * 0.1
	draw_rect(Rect2(FIELD_MARGIN, Vector2(end_zone_width, FIELD_DRAW_SIZE.y)), Color("123d8a"), true)
	draw_rect(
		Rect2(Vector2(FIELD_MARGIN.x + FIELD_DRAW_SIZE.x - end_zone_width, FIELD_MARGIN.y), Vector2(end_zone_width, FIELD_DRAW_SIZE.y)),
		Color("7b1e3a"),
		true
	)

	for yard in range(0, 101, 5):
		var x: float = field_to_screen({"x": yard, "y": 0.0}).x
		var line_width: float = 4.0 if yard % 10 == 0 else 2.0
		var alpha: float = 0.55 if yard % 10 == 0 else 0.2
		draw_line(Vector2(x, FIELD_MARGIN.y), Vector2(x, FIELD_MARGIN.y + FIELD_DRAW_SIZE.y), Color(1, 1, 1, alpha), line_width)

	for hash_row in [12.0, 41.3]:
		for yard in range(2, 99, 5):
			var hash_pos: Vector2 = field_to_screen({"x": yard, "y": hash_row})
			draw_line(hash_pos + Vector2(0, -6), hash_pos + Vector2(0, 6), Color(1, 1, 1, 0.5), 2.0)

	var center_line_y: float = field_to_screen({"x": 0.0, "y": float(field_data.get("width", 53.3)) / 2.0}).y
	draw_line(Vector2(FIELD_MARGIN.x, center_line_y), Vector2(FIELD_MARGIN.x + FIELD_DRAW_SIZE.x, center_line_y), Color(1, 1, 1, 0.08), 1.0)

	var pre_snap: Dictionary = current_snap.get("pre_snap", {}) as Dictionary
	var los: float = float(pre_snap.get("line_of_scrimmage", field_data.get("line_of_scrimmage", 60.0)))
	var los_x: float = field_to_screen({"x": los, "y": 0.0}).x
	draw_line(Vector2(los_x, FIELD_MARGIN.y), Vector2(los_x, FIELD_MARGIN.y + FIELD_DRAW_SIZE.y), Color("ff9f1c"), 5.0)

	var offense_text_pos: Vector2 = Vector2(FIELD_MARGIN.x + 16.0, FIELD_MARGIN.y + 22.0)
	draw_string(ThemeDB.fallback_font, offense_text_pos, "TOP-DOWN BROADCAST", HORIZONTAL_ALIGNMENT_LEFT, -1, 18, Color(1, 1, 1, 0.75))
