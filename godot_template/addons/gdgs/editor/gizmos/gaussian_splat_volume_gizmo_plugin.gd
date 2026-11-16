@tool
extends EditorNode3DGizmoPlugin

## Editor gizmo for GaussianSplatVolume: the shape as a wireframe plus one
## handle per face. Dragging a handle moves that face and keeps the opposite
## one where it is.

const COLOR := Color(1.0, 0.8, 0.25)
const MIN_SIZE := 0.01
const CIRCLE_STEPS := 48
const AXIS_NAMES := ["X", "Y", "Z"]

var undo_redo: EditorUndoRedoManager


func _init() -> void:
	create_material("shape", COLOR)
	create_handle_material("handles")


func _get_gizmo_name() -> String: return "GaussianSplatVolume"
func _get_priority() -> int: return 0
func _has_gizmo(node: Node3D) -> bool: return node is GaussianSplatVolume


func _redraw(gizmo: EditorNode3DGizmo) -> void:
	gizmo.clear()
	var volume := gizmo.get_node_3d() as GaussianSplatVolume
	if volume == null:
		return
	var half := volume.size * 0.5
	var lines := PackedVector3Array()
	match volume.shape:
		GaussianSplatVolume.Shape.ELLIPSOID:
			_add_ring(lines, half, 0, 1, 0.0)
			_add_ring(lines, half, 1, 2, 0.0)
			_add_ring(lines, half, 0, 2, 0.0)
		GaussianSplatVolume.Shape.CYLINDER:
			_add_ring(lines, half, 0, 2, 1.0)
			_add_ring(lines, half, 0, 2, -1.0)
			for corner in [Vector3(1, 0, 0), Vector3(-1, 0, 0), Vector3(0, 0, 1), Vector3(0, 0, -1)]:
				lines.append((corner + Vector3.UP) * half)
				lines.append((corner + Vector3.DOWN) * half)
		_:
			var box := AABB(-half, volume.size)
			for i in range(8):
				for bit in [1, 2, 4]:
					if i & bit == 0:
						lines.append(box.get_endpoint(i))
						lines.append(box.get_endpoint(i | bit))
	gizmo.add_lines(lines, get_material("shape", gizmo))
	gizmo.add_collision_segments(lines)

	var handles := PackedVector3Array()
	for id in range(6):
		handles.append(_handle_direction(id) * half)
	gizmo.add_handles(handles, get_material("handles", gizmo), [])


## A circle in the plane of axes `a` and `b`, shifted along the third axis by
## `offset` half-extents.
func _add_ring(lines: PackedVector3Array, half: Vector3, a: int, b: int, offset: float) -> void:
	var third := 3 - a - b
	for step in range(CIRCLE_STEPS):
		for k in [step, step + 1]:
			var angle := TAU * float(k) / CIRCLE_STEPS
			var point := Vector3.ZERO
			point[a] = cos(angle) * half[a]
			point[b] = sin(angle) * half[b]
			point[third] = offset * half[third]
			lines.append(point)


## Handles 0..5 are +X, -X, +Y, -Y, +Z, -Z.
func _handle_direction(id: int) -> Vector3:
	var direction := Vector3.ZERO
	direction[id / 2] = 1.0 if id % 2 == 0 else -1.0
	return direction


func _get_handle_name(_gizmo: EditorNode3DGizmo, id: int, _secondary: bool) -> String:
	return "Size " + AXIS_NAMES[id / 2]


func _get_handle_value(gizmo: EditorNode3DGizmo, _id: int, _secondary: bool) -> Variant:
	var volume := gizmo.get_node_3d() as GaussianSplatVolume
	return [volume.size, volume.global_position]


func _set_handle(gizmo: EditorNode3DGizmo, id: int, _secondary: bool, camera: Camera3D, screen_pos: Vector2) -> void:
	var volume := gizmo.get_node_3d() as GaussianSplatVolume
	var axis := id / 2
	var direction := _handle_direction(id)
	var xform := volume.global_transform
	var axis_world := xform.basis * direction  # one local metre along the handle
	var axis_scale := axis_world.length()
	if axis_scale < 1e-6:
		return
	var fixed_face := xform * (-direction * volume.size * 0.5)
	var ray_from := camera.project_ray_origin(screen_pos)
	var ray_to := ray_from + camera.project_ray_normal(screen_pos) * 4096.0
	var axis_unit := axis_world / axis_scale
	var closest := Geometry3D.get_closest_points_between_segments(
		fixed_face - axis_unit * 4096.0, fixed_face + axis_unit * 4096.0, ray_from, ray_to)
	var new_extent := maxf((closest[0] - fixed_face).dot(axis_unit) / axis_scale, MIN_SIZE)

	var new_size := volume.size
	var grown := new_extent - new_size[axis]
	new_size[axis] = new_extent
	volume.size = new_size
	volume.global_position += axis_world * grown * 0.5


func _commit_handle(gizmo: EditorNode3DGizmo, _id: int, _secondary: bool, restore: Variant, cancel: bool) -> void:
	var volume := gizmo.get_node_3d() as GaussianSplatVolume
	if cancel or undo_redo == null:
		if cancel:
			volume.size = restore[0]
			volume.global_position = restore[1]
		return
	undo_redo.create_action("Resize Splat Volume")
	undo_redo.add_do_property(volume, "size", volume.size)
	undo_redo.add_do_property(volume, "global_position", volume.global_position)
	undo_redo.add_undo_property(volume, "size", restore[0])
	undo_redo.add_undo_property(volume, "global_position", restore[1])
	undo_redo.commit_action()
