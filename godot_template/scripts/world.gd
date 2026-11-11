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

	# Optional "backdrop": what to show where the splat has gaps.
	var backdrop: Dictionary = {}
	if typeof(m.get("backdrop")) == TYPE_DICTIONARY:
		backdrop = m["backdrop"]

	var splat := SplatLoader.create(m["splat"]["path"])
	if splat:
		add_child(splat)
		var c: Array = m["splat"]["center"]
		SplatLoader.finish_setup(self, splat, Vector3(c[0], c[1], c[2]),
				_color(backdrop.get("background"), Color.BLACK))
	var floor_y: float = m["collision"].get("floor_y", 0.0)
	_add_collision(m["collision"]["path"], splat == null, splat != null and backdrop.get("mesh", false))
	_add_safety_floor(floor_y)
	if splat and backdrop.has("floor"):
		_add_backdrop_floor(backdrop["floor"], floor_y)

	var spawn: Dictionary = m["spawn"]
	var p: Array = spawn["position"]
	player.set_spawn(Vector3(p[0], p[1], p[2]), deg_to_rad(spawn["yaw_deg"]))
	player.set_eye_height(m.get("eye_height", 1.6))

	# m["objects"] is reserved for interactable objects in a later version.

	# Automated check: R2V_SCREENSHOT=<png path> saves one frame and quits.
	var shot := OS.get_environment("R2V_SCREENSHOT")
	if shot != "":
		_screenshot_and_quit(shot)


func _color(value: Variant, fallback: Color) -> Color:
	# Colours in scene.json are sRGB 0-1 triples.
	if typeof(value) != TYPE_ARRAY or value.size() < 3:
		return fallback
	return Color(value[0], value[1], value[2])


func _backdrop_material() -> StandardMaterial3D:
	# Unlit: the colours already contain the room's lighting.
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	return mat


func _add_backdrop_floor(spec: Dictionary, floor_y: float) -> void:
	# A plain floor in the room's floor colour, a little below the scanned one, so
	# unfilmed floor does not show as a hole under the player's feet.
	var lo: Array = spec["min"]
	var hi: Array = spec["max"]
	var plane := PlaneMesh.new()
	plane.size = Vector2(hi[0] - lo[0], hi[1] - lo[1])
	var mat := _backdrop_material()
	mat.albedo_color = _color(spec.get("color"), Color(0.3, 0.3, 0.3))
	var node := MeshInstance3D.new()
	node.name = "BackdropFloor"
	node.mesh = plane
	node.material_override = mat
	node.position = Vector3((lo[0] + hi[0]) / 2.0, floor_y - 0.05, (lo[1] + hi[1]) / 2.0)
	add_child(node)


func _add_collision(path: String, show_mesh: bool, as_backdrop: bool) -> void:
	var packed: PackedScene = load(path)
	if packed == null:
		push_error("collision mesh not found: " + path)
		return
	var root := packed.instantiate()
	add_child(root)
	for node in root.find_children("*", "MeshInstance3D", true, false):
		var mesh := node as MeshInstance3D
		mesh.create_trimesh_collision()
		mesh.visible = show_mesh or as_backdrop
		if as_backdrop:
			# Shown behind the splat in its own colours: gaps in the splat then show
			# a plain version of the surface instead of the background.
			var mat := _backdrop_material()
			mat.vertex_color_use_as_albedo = true
			mat.vertex_color_is_srgb = true
			mesh.material_override = mat
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
	# R2V_SHOT_YAW=<degrees> turns the player first, to check other directions.
	var turn := OS.get_environment("R2V_SHOT_YAW")
	if turn != "":
		player.rotation.y += deg_to_rad(float(turn))
	for i in 90:  # let the splat upload and the player settle on the floor
		await get_tree().process_frame
	get_viewport().get_texture().get_image().save_png(path)
	print("screenshot saved: %s  player at %s  on_floor=%s" % [path, player.position, player.is_on_floor()])
	get_tree().quit()
