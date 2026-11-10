class_name SplatLoader
## The only file that knows about the Gaussian splat addon (GDGS 3.3.0, MIT,
## vendored in res://addons/gdgs).


## Returns a splat node for an imported .splat/.ply, or null if it cannot load
## (for example the project was never opened in the editor, so nothing is imported).
static func create(path: String) -> Node3D:
	if not ResourceLoader.exists(path):
		push_warning("Splat not imported: %s. Open the project in the Godot editor once." % path)
		return null
	var res: Resource = load(path)
	if not (res is GaussianResource):
		push_warning("Not a Gaussian splat resource: " + path)
		return null
	var node := GaussianSplatNode.new()
	node.gaussian = res
	return node


## Call after the node is in the tree.
## `center` is splat.center from scene.json. `background` is shown where nothing was filmed.
static func finish_setup(world: Node3D, node: Node3D, center: Vector3, background: Color) -> void:
	# GDGS rotates identity-transform nodes by 180 degrees to stand Y-down captures
	# upright. The r2v pipeline already outputs Y-up, so undo that.
	# GDGS also recentres imported data on its mean position; put it back.
	node.transform = Transform3D(Basis.IDENTITY, center)

	# The compute backend draws through a compositor pass.
	var effect := GaussianCompositorEffect.new()
	# The backdrop mesh and floor may only show through gaps in the splat, never
	# hide it: the mesh is wrong at windows and glass, where it would cover the
	# view outside. A huge bias switches the compositor's depth rejection off.
	effect.depth_bias = 100000.0
	# With scene geometry behind it, the compositor drops splat pixels that have no
	# depth, and depth is only recorded above this opacity. Keep faint splats too.
	effect.depth_capture_alpha = 0.02
	var compositor := Compositor.new()
	compositor.compositor_effects = [effect]
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = background
	var world_env := WorldEnvironment.new()
	world_env.environment = env
	world_env.compositor = compositor
	world.add_child(world_env)
