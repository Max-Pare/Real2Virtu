extends CharacterBody3D
## First-person controller with two modes.
## Walk: WASD or arrows, mouse look, Space jumps, gravity and collision on.
## Fly (toggle with F): WASD moves where the camera looks, Space/E up, Q/Ctrl down,
##   no gravity, passes through everything.
## Shift moves faster in both modes. Esc frees the mouse, click captures it again.

const WALK_SPEED := 2.5
const FLY_SPEED := 3.0
const FAST_FACTOR := 3.0
const JUMP_SPEED := 3.5
const MOUSE_SENS := 0.0025

@onready var camera: Camera3D = $Camera
@onready var body_shape: CollisionShape3D = $Collision

var flying := false
var _gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")


func _ready() -> void:
	_bind("move_forward", [KEY_W, KEY_UP])
	_bind("move_back", [KEY_S, KEY_DOWN])
	_bind("move_left", [KEY_A, KEY_LEFT])
	_bind("move_right", [KEY_D, KEY_RIGHT])
	_bind("jump", [KEY_SPACE])
	_bind("fly_up", [KEY_SPACE, KEY_E])
	_bind("fly_down", [KEY_Q, KEY_CTRL])
	_bind("move_fast", [KEY_SHIFT])
	_bind("toggle_fly", [KEY_F])
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func set_eye_height(h: float) -> void:
	camera.position.y = h


func set_flying(on: bool) -> void:
	flying = on
	body_shape.disabled = on  # fly mode passes through walls
	velocity = Vector3.ZERO


func _bind(action: String, keys: Array) -> void:
	if InputMap.has_action(action):
		return
	InputMap.add_action(action)
	for key in keys:
		var ev := InputEventKey.new()
		ev.physical_keycode = key
		InputMap.action_add_event(action, ev)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * MOUSE_SENS)
		camera.rotation.x = clamp(camera.rotation.x - event.relative.y * MOUSE_SENS, -1.5, 1.5)
	elif event.is_action_pressed("toggle_fly"):
		set_flying(not flying)
	elif event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	elif event is InputEventMouseButton and event.pressed:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _physics_process(delta: float) -> void:
	var input := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var fast := FAST_FACTOR if Input.is_action_pressed("move_fast") else 1.0
	if flying:
		# Move along the camera's own axes, so looking up and pressing W climbs.
		var dir := camera.global_transform.basis * Vector3(input.x, 0, input.y)
		dir.y += Input.get_axis("fly_down", "fly_up")
		position += dir.limit_length(1.0) * FLY_SPEED * fast * delta
		return

	if not is_on_floor():
		velocity.y -= _gravity * delta
	elif Input.is_action_just_pressed("jump"):
		velocity.y = JUMP_SPEED
	var dir := (transform.basis * Vector3(input.x, 0, input.y)).normalized()
	velocity.x = dir.x * WALK_SPEED * fast
	velocity.z = dir.z * WALK_SPEED * fast
	move_and_slide()
