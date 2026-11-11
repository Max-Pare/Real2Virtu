extends SceneTree
## Headless check that the player does not get stuck.
##
## Copy into an exported project as res://dev/walk_test.gd and run:
##   godot --headless --fixed-fps 60 --path <project> --script res://dev/walk_test.gd
##
## The player walks in a random direction for one second, again and again. A wall
## stops it in some directions, which is fine. When eight tries in a row go nowhere,
## 16 evenly spaced directions are probed from that spot: "stuck" means none of them
## moves the player; otherwise it was only a tight spot. The player is then put back
## at the spawn point and the walk goes on.
## R2V_WALK_TRIES=<n> overrides the number of tries.

const EPISODES := 2000
const EPISODE_FRAMES := 60
const TRIES_BEFORE_STUCK := 8

var _scene: Node


func _initialize() -> void:
	_scene = load("res://main.tscn").instantiate()
	root.add_child(_scene)
	_walk()


## Scans have open sides. Fence the scanned area in, so the walk stays among the
## geometry instead of wandering off across the empty safety floor.
func _fence() -> void:
	var box := AABB()
	var first := true
	for node in _scene.find_children("*", "MeshInstance3D", true, false):
		var mi := node as MeshInstance3D
		var b: AABB = mi.global_transform * mi.get_aabb()
		box = b if first else box.merge(b)
		first = false
	if first:
		return
	var lo := box.position - Vector3(0.3, 0, 0.3)
	var hi := box.end + Vector3(0.3, 0, 0.3)
	var planes := [Plane(Vector3.RIGHT, lo.x), Plane(Vector3.LEFT, -hi.x),
			Plane(Vector3.BACK, lo.z), Plane(Vector3.FORWARD, -hi.z)]
	for p in planes:
		var shape := CollisionShape3D.new()
		var boundary := WorldBoundaryShape3D.new()
		boundary.plane = p
		shape.shape = boundary
		var body := StaticBody3D.new()
		body.add_child(shape)
		_scene.add_child(body)
	print("WALKTEST area %.1f x %.1f m" % [hi.x - lo.x, hi.z - lo.z])


func _walk() -> void:
	var player: CharacterBody3D = _scene.get_node("Player")
	for i in 5:
		await physics_frame
	_fence()
	for i in 60:
		await physics_frame
	var spawn := player.position
	var rng := RandomNumberGenerator.new()
	rng.seed = 1
	var stuck: Array[Vector3] = []
	var fails := 0
	var blocked := 0
	var tight := 0
	var travelled := 0.0
	var reach := 0.0
	var episodes := EPISODES
	if OS.get_environment("R2V_WALK_TRIES") != "":
		episodes = int(OS.get_environment("R2V_WALK_TRIES"))
	var started := Time.get_ticks_msec()
	for e in episodes:
		if e % 100 == 0:
			print("WALKTEST progress %d/%d  %.0fs  at (%.2f, %.2f, %.2f)" % [
				e, episodes, (Time.get_ticks_msec() - started) / 1000.0,
				player.position.x, player.position.y, player.position.z])
		player.rotation.y = rng.randf_range(0.0, TAU)
		player.input_override = Vector2(0, -1)
		var before := player.position
		for i in EPISODE_FRAMES:
			await physics_frame
		var moved := Vector2(player.position.x - before.x, player.position.z - before.z).length()
		travelled += moved
		reach = maxf(reach, Vector2(player.position.x - spawn.x, player.position.z - spawn.z).length())
		if moved < 0.05:
			fails += 1
			blocked += 1
		else:
			fails = 0
		if fails >= TRIES_BEFORE_STUCK:
			# Suspicion only: a pocket with a narrow exit also fails random tries.
			# Probe 16 evenly spaced directions from this exact spot.
			var here := player.position
			var free_dirs := 0
			for k in 16:
				player.position = here
				player.velocity = Vector3.ZERO
				player.rotation.y = TAU * k / 16.0
				for i in 45:
					await physics_frame
				if Vector2(player.position.x - here.x, player.position.z - here.z).length() >= 0.1:
					free_dirs += 1
			if free_dirs == 0:
				stuck.append(here)
			else:
				tight += 1
			player.position = spawn
			player.velocity = Vector3.ZERO
			fails = 0
			for i in 10:
				await physics_frame
	print("WALKTEST tries=%d blocked=%d tight_spots=%d stuck=%d self_freed=%d travelled=%.1fm reach=%.1fm" % [
		episodes, blocked, tight, stuck.size(), player.unstuck_count, travelled, reach])
	for s in stuck:
		print("WALKTEST stuck at (%.2f, %.2f, %.2f)" % [s.x, s.y, s.z])
	quit()
