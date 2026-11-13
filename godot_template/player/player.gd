extends CharacterBody3D
## First-person controller with two modes.
## Walk: WASD or arrows, mouse look, Space jumps, gravity and collision on.
## Fly (toggle with F): WASD moves where the camera looks, Space/E up, Q/Ctrl down,
##   no gravity, passes through everything.
## Shift moves faster in both modes. R returns to the start point.
## Mouse wheel zooms by narrowing the camera's field of view; middle click resets it.
## Esc frees the mouse, click captures it again.
##
## Scanned collision meshes are rough, so the walker can end up wedged. When it is
## blocked in every direction it frees itself: a small step up if that clears the
## obstacle, otherwise back to the last place where walking worked.

const STUCK_AFTER := 0.4      # seconds of pushing without moving before checking
const SAFE_SPOT_EVERY := 0.5  # seconds of free walking between remembered spots
const SAFE_SPOTS_KEPT := 10
const STEP_HEIGHTS := [0.1, 0.2, 0.3, 0.45]
const PROBE := 0.15           # metres tested in each direction

const WALK_SPEED := 2.5
const FLY_SPEED := 3.0
const FAST_FACTOR := 3.0
const JUMP_SPEED := 3.5
const MOUSE_SENS := 0.0025
const ZOOM_STEP := 0.85       # field of view is multiplied by this per wheel notch
const MIN_FOV := 5.0          # degrees; the widest is the camera's own setting

@onready var camera: Camera3D = $Camera
@onready var body_shape: CollisionShape3D = $Collision

var flying := false
var _base_fov := 75.0
## Set to a Vector2 to drive the player from a script instead of the keyboard (automated tests).
var input_override: Variant = null
## How often the walker had to free itself (read by automated tests).
var unstuck_count := 0
var _stuck_for := 0.0
var _free_for := 0.0
var _safe_spots: Array[Vector3] = []
var _spawn := Transform3D.IDENTITY
var _has_spawn := false
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
	_bind("respawn", [KEY_R])
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	_base_fov = camera.fov


func set_eye_height(h: float) -> void:
	camera.position.y = h


## Remember where the level starts; R comes back here.
func set_spawn(pos: Vector3, yaw: float) -> void:
	position = pos
	rotation.y = yaw
	_spawn = transform
	_has_spawn = true
	_safe_spots.clear()


func respawn() -> void:
	if not _has_spawn:
		return
	transform = _spawn
	camera.rotation.x = 0.0
	velocity = Vector3.ZERO
	_stuck_for = 0.0
	_safe_spots.clear()


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
		# Turn slower when zoomed in, so the picture moves at the same speed on screen.
		var sens := MOUSE_SENS * camera.fov / _base_fov
		rotate_y(-event.relative.x * sens)
		camera.rotation.x = clamp(camera.rotation.x - event.relative.y * sens, -1.5, 1.5)
	elif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_WHEEL_UP:
		camera.fov = clampf(camera.fov * ZOOM_STEP, MIN_FOV, _base_fov)
	elif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
		camera.fov = clampf(camera.fov / ZOOM_STEP, MIN_FOV, _base_fov)
	elif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_MIDDLE:
		camera.fov = _base_fov
	elif event.is_action_pressed("toggle_fly"):
		set_flying(not flying)
	elif event.is_action_pressed("respawn"):
		respawn()
	elif event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	elif event is InputEventMouseButton and event.pressed:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _physics_process(delta: float) -> void:
	var input := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	if input_override != null:
		input = input_override
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
	var before := position
	move_and_slide()
	_watch_for_stuck(dir, Vector2(position.x - before.x, position.z - before.z).length(), delta)


func _watch_for_stuck(dir: Vector3, moved: float, delta: float) -> void:
	if dir.length() < 0.1:
		_stuck_for = 0.0
		return
	if moved > 0.2 * WALK_SPEED * delta:
		_stuck_for = 0.0
		_free_for += delta
		if _free_for >= SAFE_SPOT_EVERY and is_on_floor():
			_free_for = 0.0
			_safe_spots.append(position)
			if _safe_spots.size() > SAFE_SPOTS_KEPT:
				_safe_spots.pop_front()
		return
	_stuck_for += delta
	if _stuck_for < STUCK_AFTER:
		return
	_stuck_for = 0.0
	# Pushing against a wall also goes nowhere. Only act when no direction is open.
	if _any_direction_open(global_transform):
		return
	_free_self(dir)


func _any_direction_open(from: Transform3D) -> bool:
	for k in 8:
		var probe := Vector3(cos(TAU * k / 8.0), 0, sin(TAU * k / 8.0)) * PROBE
		if not test_move(from, probe):
			return true
	return false


func _free_self(dir: Vector3) -> void:
	unstuck_count += 1
	# 1. A small step up, if the way the player wants to go is open from there.
	for h: float in STEP_HEIGHTS:
		var up: Vector3 = Vector3.UP * h
		if test_move(global_transform, up):
			break  # something overhead; higher steps are blocked too
		if not test_move(global_transform.translated(up), dir * PROBE):
			position += up + dir * PROBE
			velocity = Vector3.ZERO
			return
	# 2. Back to the most recent place where walking worked.
	while not _safe_spots.is_empty():
		var spot: Vector3 = _safe_spots.pop_back()
		if spot.distance_to(position) < 0.3:
			continue  # too close to the trap
		position = spot
		velocity = Vector3.ZERO
		return
	# 3. Nothing remembered: start over.
	respawn()
