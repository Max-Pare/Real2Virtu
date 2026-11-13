"""Interior video to walkable Godot map."""

from .gsplat_compat import point_to_gpu_id_table, use_system_rocm_runtime

# Both must happen before torch is imported.
point_to_gpu_id_table()
use_system_rocm_runtime()
