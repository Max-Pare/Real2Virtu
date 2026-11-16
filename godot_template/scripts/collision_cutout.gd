class_name CollisionCutout
## Divides the room's collision mesh between the room and the SplatObjects.
##
## A triangle belongs to the first object (in tree order) whose shape holds its
## centre, the same rule the splat shader uses for Gaussians. The object's
## triangles leave the room's visible mesh and become a child of the object, so
## the plain mesh behind the splat moves with it. What happens to their
## collision is the object's `collision` setting.


## Rebuilds `mesh` without the objects' triangles, gives it a static collider,
## and hands each object its piece. Returns the pieces (for the caller to style
## like the room mesh). Call when the objects are at their rest pose.
static func split(mesh: MeshInstance3D, objects: Array[SplatObject]) -> Array[MeshInstance3D]:
	var pieces: Array[MeshInstance3D] = []
	var source := mesh.mesh
	if source == null:
		return pieces
	var to_world := mesh.global_transform
	var to_unit: Array[Transform3D] = []
	for object in objects:
		to_unit.append(object.get_rest_to_unit() * to_world)

	var room_mesh := ArrayMesh.new()
	var room_faces := PackedVector3Array()  # the room's collision, in mesh space
	var object_meshes: Array[ArrayMesh] = []
	var object_faces: Array[PackedVector3Array] = []
	for object in objects:
		object_meshes.append(ArrayMesh.new())
		object_faces.append(PackedVector3Array())

	for surface in source.get_surface_count():
		if source.surface_get_primitive_type(surface) != Mesh.PRIMITIVE_TRIANGLES:
			continue
		var arrays := source.surface_get_arrays(surface)
		var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
		var indices := PackedInt32Array()
		if arrays[Mesh.ARRAY_INDEX] != null:
			indices = arrays[Mesh.ARRAY_INDEX]
		else:
			indices = PackedInt32Array(range(vertices.size()))

		var room_indices := PackedInt32Array()
		var object_indices: Array[PackedInt32Array] = []
		for object in objects:
			object_indices.append(PackedInt32Array())

		for t in range(0, indices.size() - 2, 3):
			var a := vertices[indices[t]]
			var b := vertices[indices[t + 1]]
			var c := vertices[indices[t + 2]]
			var centre := (a + b + c) / 3.0
			var owner := -1
			for i in objects.size():
				if GaussianSplatVolume.unit_contains(objects[i].shape, to_unit[i] * centre):
					owner = i
					break
			if owner < 0:
				room_indices.append_array([indices[t], indices[t + 1], indices[t + 2]])
				room_faces.append_array([a, b, c])
				continue
			object_indices[owner].append_array([indices[t], indices[t + 1], indices[t + 2]])
			if objects[owner].collision == SplatObject.Collision.STAYS:
				room_faces.append_array([a, b, c])
			else:
				var to_object := objects[owner].global_transform.affine_inverse() * to_world
				object_faces[owner].append_array([to_object * a, to_object * b, to_object * c])

		var material := source.surface_get_material(surface)
		if not room_indices.is_empty():
			arrays[Mesh.ARRAY_INDEX] = room_indices
			room_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
			room_mesh.surface_set_material(room_mesh.get_surface_count() - 1, material)
		for i in objects.size():
			if object_indices[i].is_empty():
				continue
			var to_object := objects[i].global_transform.affine_inverse() * to_world
			var piece := object_meshes[i]
			piece.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES,
					_extract(arrays, object_indices[i], to_object))
			piece.surface_set_material(piece.get_surface_count() - 1, material)

	mesh.mesh = room_mesh
	if not room_faces.is_empty():
		var shape := ConcavePolygonShape3D.new()
		shape.set_faces(room_faces)
		var holder := CollisionShape3D.new()
		holder.shape = shape
		var body := StaticBody3D.new()
		body.add_child(holder)
		mesh.add_child(body)

	for i in objects.size():
		if object_meshes[i].get_surface_count() == 0:
			continue
		var piece := MeshInstance3D.new()
		piece.name = "RoomPiece"
		piece.mesh = object_meshes[i]
		objects[i].adopt_room_piece(piece, object_faces[i])
		pieces.append(piece)
	return pieces


## The triangles in `indices` as a surface of their own, moved by `xform`.
## Keeps positions, normals and vertex colours.
static func _extract(arrays: Array, indices: PackedInt32Array, xform: Transform3D) -> Array:
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var has_normals: bool = arrays[Mesh.ARRAY_NORMAL] != null
	var has_colors: bool = arrays[Mesh.ARRAY_COLOR] != null
	var normals := PackedVector3Array()
	var colors := PackedColorArray()
	if has_normals:
		normals = arrays[Mesh.ARRAY_NORMAL]
	if has_colors:
		colors = arrays[Mesh.ARRAY_COLOR]

	var remap := {}
	var out_vertices := PackedVector3Array()
	var out_normals := PackedVector3Array()
	var out_colors := PackedColorArray()
	var out_indices := PackedInt32Array()
	for index in indices:
		if not remap.has(index):
			remap[index] = out_vertices.size()
			out_vertices.append(xform * vertices[index])
			if has_normals:
				out_normals.append((xform.basis * normals[index]).normalized())
			if has_colors:
				out_colors.append(colors[index])
		out_indices.append(remap[index])

	var out := []
	out.resize(Mesh.ARRAY_MAX)
	out[Mesh.ARRAY_VERTEX] = out_vertices
	out[Mesh.ARRAY_INDEX] = out_indices
	if has_normals:
		out[Mesh.ARRAY_NORMAL] = out_normals
	if has_colors:
		out[Mesh.ARRAY_COLOR] = out_colors
	return out
