extends Node3D
## Builds the level from scene.json, the manifest written by the r2v pipeline.

const MANIFEST := "res://scene.json"

@onready var player: CharacterBody3D = $Player


func _ready() -> void:
	var text := FileAccess.get_file_as_string(MANIFEST)
	var m: Variant = JSON.parse_string(text)
	if typeof(m) != TYPE_DICTIONARY:
		push_error("scene.json missing or invalid; run the r2v pipeline export stage")
		return

	var splat := SplatLoader.create(m["splat"]["path"])
	if splat:
		add_child(splat)
		var c: Array = m["splat"]["center"]
		SplatLoader.finish_setup(self, splat, Vector3(c[0], c[1], c[2]))
	_add_collision(m["collision"]["path"], splat == null)
	_add_safety_floor(m["collision"].get("floor_y", 0.0))

	var spawn: Dictionary = m["spawn"]
	var p: Array = spawn["position"]
	player.position = Vector3(p[0], p[1], p[2])
	player.rotation.y = deg_to_rad(spawn["yaw_deg"])
	player.set_eye_height(m.get("eye_height", 1.6))

	# m["objects"] is reserved for interactable objects in a later version.

	# Automated check: R2V_SCREENSHOT=<png path> saves one frame and quits.
	var shot := OS.get_environment("R2V_SCREENSHOT")
	if shot != "":
		_screenshot_and_quit(shot)


func _add_collision(path: String, show_mesh: bool) -> void:
	var packed: PackedScene = load(path)
	if packed == null:
		push_error("collision mesh not found: " + path)
		return
	var root := packed.instantiate()
	add_child(root)
	for node in root.find_children("*", "MeshInstance3D", true, false):
		var mesh := node as MeshInstance3D
		mesh.create_trimesh_collision()
		mesh.visible = show_mesh
	if show_mesh:
		# No splat available: light the grey collision mesh so the map is still walkable.
		var sun := DirectionalLight3D.new()
		sun.rotation_degrees = Vector3(-55, 30, 0)
		add_child(sun)
		var env := WorldEnvironment.new()
		env.environment = Environment.new()
		env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
		env.environment.ambient_light_color = Color(0.6, 0.6, 0.6)
		add_child(env)


func _add_safety_floor(floor_y: float) -> void:
	# The scanned floor has holes wherever it was never filmed (often right under
	# the camera path). An invisible flat plane just below it stops the player falling through.
	var shape := CollisionShape3D.new()
	shape.shape = WorldBoundaryShape3D.new()
	var body := StaticBody3D.new()
	body.name = "SafetyFloor"
	body.position.y = floor_y - 0.02
	body.add_child(shape)
	add_child(body)


func _screenshot_and_quit(path: String) -> void:
	# Automated run: leave the user's mouse alone and ignore their input.
	Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	player.set_process_unhandled_input(false)
	for i in 90:  # let the splat upload and the player settle on the floor
		await get_tree().process_frame
	get_viewport().get_texture().get_image().save_png(path)
	print("screenshot saved: %s  player at %s  on_floor=%s" % [path, player.position, player.is_on_floor()])
	get_tree().quit()
