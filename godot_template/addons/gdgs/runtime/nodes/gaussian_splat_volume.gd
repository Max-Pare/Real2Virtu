@tool
@icon("res://addons/gdgs/editor/icons/gaussian_splat_volume.svg")
extends Node3D
class_name GaussianSplatVolume

## Claims every Gaussian whose centre lies inside this shape and lets the node
## move, tint and fade them as one object.
##
## Nothing is copied or cut out of the splat data: the projection shader tests
## each Gaussian against the active volumes every frame. The Gaussians are
## claimed where the volume stands at rest (its transform when the game starts)
## and follow the node from there, so moving, rotating or animating the node
## moves the object. In the editor the rest pose is always the current pose, so
## moving the node there moves the selection instead.
##
## Works on any GaussianSplatNode in the scene, in world space. Where volumes
## overlap, the first one in the tree wins. Compute backend only.

enum Shape { BOX, ELLIPSOID, CYLINDER }

const SELECTOR_SCRIPT := preload("res://addons/gdgs/runtime/render/backend/gaussian_backend_selector.gd")

## Must match MAX_VOLUMES and the SplatVolume struct in gsplat_projection.glsl.
const MAX_VOLUMES := 32
const FLOATS_PER_VOLUME := 40
const HEADER_FLOATS := 4
## Tint shown in the editor so the claimed Gaussians can be seen.
const EDITOR_PREVIEW_STRENGTH := 0.55

@export var shape: Shape = Shape.BOX:
	set(value):
		shape = value
		update_gizmos()
## Full extent along each local axis, in metres. A cylinder stands along Y.
@export var size := Vector3(0.5, 0.5, 0.5):
	set(value):
		size = value.max(Vector3(0.001, 0.001, 0.001))
		update_gizmos()

@export_group("Appearance")
@export var highlight_color := Color(1.0, 0.8, 0.25)
## 0 leaves the Gaussians as scanned, 1 paints them in the highlight colour.
@export_range(0.0, 1.0, 0.01) var highlight_strength := 0.0
## Fades the claimed Gaussians out. A hidden node counts as fully faded.
@export_range(0.0, 1.0, 0.01) var opacity := 1.0
## Tint the claimed Gaussians in the editor. Never shown in the game.
@export var editor_preview := true

static var _volumes: Array[GaussianSplatVolume] = []
static var _warned_overflow := false
static var _checked_backend := false

var _rest := Transform3D.IDENTITY
var _has_rest := false


func _enter_tree() -> void:
	if not _volumes.has(self):
		_volumes.append(self)


func _exit_tree() -> void:
	_volumes.erase(self)


func _ready() -> void:
	if not Engine.is_editor_hint():
		capture_rest()
	_check_backend.call_deferred()


func _check_backend() -> void:
	if _checked_backend or not is_inside_tree():
		return
	_checked_backend = true
	var backend: RefCounted = SELECTOR_SCRIPT.get_backend(self)
	if backend != null and backend.get_display_name() != "Compute":
		push_warning("[gdgs] splat volumes need the Compute backend; they have no effect on %s"
				% backend.get_display_name())


## Claims the Gaussians at the current pose. Called once when the game starts;
## call it again to let go of the current ones and claim what is here now.
func capture_rest() -> void:
	_rest = global_transform
	_has_rest = true


## The pose at which the Gaussians were claimed.
func get_rest_transform() -> Transform3D:
	if _has_rest and not Engine.is_editor_hint():
		return _rest
	return global_transform


## Maps world space onto the unit shape at the rest pose: the box becomes the
## cube from -1 to 1, the ellipsoid the unit sphere, the cylinder radius 1 and
## half-height 1. The shader uses the same matrix.
func get_rest_to_unit() -> Transform3D:
	return Transform3D(Basis.from_scale(Vector3(2.0, 2.0, 2.0) / size), Vector3.ZERO) \
			* get_rest_transform().affine_inverse()


## Whether a point in unit-shape space is inside. Parity with the volume test
## in gsplat_projection.glsl.
static func unit_contains(shape_kind: Shape, u: Vector3) -> bool:
	match shape_kind:
		Shape.ELLIPSOID:
			return u.length_squared() <= 1.0
		Shape.CYLINDER:
			return u.x * u.x + u.z * u.z <= 1.0 and absf(u.y) <= 1.0
	return absf(u.x) <= 1.0 and absf(u.y) <= 1.0 and absf(u.z) <= 1.0


## World-space distance from a point to the volume's surface at its current
## pose; 0 inside. Uses the bounding box for every shape.
func distance_to(point: Vector3) -> float:
	var local := global_transform.affine_inverse() * point
	var half := size * 0.5
	var nearest := local.clamp(-half, half)
	return (global_transform * nearest).distance_to(point)


## Shader-ready block for every volume in the tree: a vec4 header (x = count)
## followed by MAX_VOLUMES fixed slots. Main thread only.
static func pack_all() -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(HEADER_FLOATS + MAX_VOLUMES * FLOATS_PER_VOLUME)
	var count := 0
	for volume in _volumes:
		if not is_instance_valid(volume) or not volume.is_inside_tree():
			continue
		if count >= MAX_VOLUMES:
			if not _warned_overflow:
				_warned_overflow = true
				push_warning("[gdgs] more than %d splat volumes; the rest are ignored" % MAX_VOLUMES)
			break
		volume._pack(out, HEADER_FLOATS + count * FLOATS_PER_VOLUME)
		count += 1
	out[0] = float(count)
	return out


func _pack(out: PackedFloat32Array, base: int) -> void:
	var to_unit := get_rest_to_unit()
	var delta := global_transform * get_rest_transform().affine_inverse()
	_write_transform(out, base, to_unit)
	_write_transform(out, base + 16, delta)

	var strength := highlight_strength
	if Engine.is_editor_hint() and editor_preview:
		strength = maxf(strength, EDITOR_PREVIEW_STRENGTH)
	out[base + 32] = highlight_color.r
	out[base + 33] = highlight_color.g
	out[base + 34] = highlight_color.b
	out[base + 35] = clampf(strength, 0.0, 1.0)
	out[base + 36] = float(shape)
	out[base + 37] = clampf(opacity, 0.0, 1.0) if is_visible_in_tree() else 0.0


static func _write_transform(out: PackedFloat32Array, base: int, t: Transform3D) -> void:
	# Column-major mat4.
	out[base + 0] = t.basis.x.x
	out[base + 1] = t.basis.x.y
	out[base + 2] = t.basis.x.z
	out[base + 4] = t.basis.y.x
	out[base + 5] = t.basis.y.y
	out[base + 6] = t.basis.y.z
	out[base + 8] = t.basis.z.x
	out[base + 9] = t.basis.z.y
	out[base + 10] = t.basis.z.z
	out[base + 12] = t.origin.x
	out[base + 13] = t.origin.y
	out[base + 14] = t.origin.z
	out[base + 15] = 1.0
