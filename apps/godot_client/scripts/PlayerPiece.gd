class_name ReplayPlayerPiece
extends Node2D

var actor_data: Dictionary = {}

@onready var label_node: Label = $Label


func configure(payload: Dictionary) -> void:
	actor_data = payload.duplicate(true)
	var palette: Dictionary = actor_data.get("palette_tokens", {}) as Dictionary
	label_node.text = _short_label(str(actor_data.get("label", "")))
	label_node.modulate = Color(str(palette.get("secondary", "#ffffff")))
	scale = Vector2.ONE * float(actor_data.get("scale", 1.0))
	queue_redraw()


func set_screen_position(screen_position: Vector2) -> void:
	global_position = screen_position


func _draw() -> void:
	var archetype: String = str(actor_data.get("sprite_archetype_key", "toy_skill"))
	if archetype == "football":
		_draw_ball()
		return

	var palette: Dictionary = actor_data.get("palette_tokens", {}) as Dictionary
	var primary: Color = Color(str(palette.get("primary", "#204E78")))
	var secondary: Color = Color(str(palette.get("secondary", "#E0F2FE")))
	var accent: Color = Color(str(palette.get("accent", "#F6AE2D")))
	var outline: Color = Color(str(palette.get("outline", "#0B132B")))
	var heavy: bool = archetype in ["toy_heavy", "toy_front_seven"]
	var db: bool = archetype == "toy_db"

	draw_circle(Vector2(0, 10), 18 if heavy else 16, Color(0, 0, 0, 0.18))
	draw_circle(Vector2(0, 6), 16 if heavy else 14, primary.darkened(0.25))
	draw_rect(Rect2(Vector2(-6 if heavy else -5, 18), Vector2(5, 12 if heavy else 10)), outline, true)
	draw_rect(Rect2(Vector2(1, 18), Vector2(5, 12 if heavy else 10)), outline, true)
	draw_rect(Rect2(Vector2(-10 if heavy else -8, -6), Vector2(20 if heavy else 16, 20 if heavy else 18)), primary, true)
	draw_circle(Vector2(0, -12), 12 if db else 13 if heavy else 12, secondary)
	draw_circle(Vector2(0, -15), 14 if heavy else 13, accent)
	draw_circle(Vector2(0, -15), 14 if heavy else 13, outline, false, 2.0, true)
	draw_rect(Rect2(Vector2(-9 if heavy else -8, -2), Vector2(18 if heavy else 16, 4)), outline, false, 2.0)
	draw_rect(Rect2(Vector2(-10 if heavy else -8, 2), Vector2(5, 8)), secondary, true)
	draw_rect(Rect2(Vector2(5, 2), Vector2(5, 8)), secondary, true)
	draw_circle(Vector2(0, 2), 5 if heavy else 4, accent)


func _draw_ball() -> void:
	var brown: Color = Color("#8D5524")
	var lace: Color = Color("#F8E9D2")
	draw_circle(Vector2.ZERO, 8, brown)
	draw_arc(Vector2.ZERO, 8, -1.2, 1.2, 16, lace, 1.5)
	draw_line(Vector2(-6, 0), Vector2(6, 0), lace, 1.5)
	draw_line(Vector2(0, -4), Vector2(0, 4), lace, 1.5)


func _short_label(name_text: String) -> String:
	if name_text.is_empty():
		return ""
	var chunks: PackedStringArray = name_text.split(" ", false)
	if chunks.size() == 1:
		return chunks[0].substr(0, mini(2, chunks[0].length())).to_upper()
	return (chunks[0].substr(0, 1) + chunks[chunks.size() - 1].substr(0, 1)).to_upper()
