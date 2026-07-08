"""
Blender script: Add armature (2 bones) to the Z-Anatomy arm GLB and export with skinning.

Usage:
    blender --background --python add_armature.py

This script:
1. Imports the arm model GLB
2. Creates a 2-bone armature (upper_arm + forearm) at scene origin
3. Clears mesh parent transforms (keep world positions) to flatten hierarchy
4. Deletes Empty nodes (no longer needed)
5. Parents all arm meshes to the armature with automatic weights
6. Exports new GLB with skinning data

Key design decisions:
- Armature at origin (0,0,0) with identity transform — no offset
- All meshes at scene root level (Empties removed) — consistent coordinate space
- export_yup=False keeps Z-up, matching the original arm_model.glb convention
"""

import bpy
import mathutils
from mathutils import Vector

# ===== CONFIG =====
INPUT_GLB  = "D:/shuzinuansheng/assets/3d_models/arm_model.glb"
OUTPUT_GLB = "D:/shuzinuansheng/assets/3d_models/arm_skinned.glb"

# Bone positions in Blender world space (Z-up).
# armRoot translation (0, 0, 0.7023) is already baked into the GLB's node hierarchy.
# These are the WORLD positions where bones should be placed.
SHOULDER = Vector((-0.197, 1.405, -0.031 + 0.7023))
ELBOW    = Vector((-0.197, 1.087, -0.031 + 0.7023))
WRIST    = Vector((-0.197, 0.856, -0.031 + 0.7023))
# ==================


def clear_scene():
    """Remove default cube, camera, light."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_glb(filepath):
    """Import GLB and return mesh objects and empty objects."""
    print(f"Importing {filepath}...")
    bpy.ops.import_scene.gltf(filepath=filepath)
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    empties = [obj for obj in bpy.context.selected_objects if obj.type == 'EMPTY']
    print(f"Found {len(meshes)} meshes, {len(empties)} empties")
    return meshes, empties


def create_armature():
    """Create a 2-bone armature at the correct world positions, at scene origin."""
    # Set 3D cursor to origin so armature is created without offset
    bpy.context.scene.cursor.location = (0, 0, 0)

    arm_data = bpy.data.armatures.new(name="arm_armature")
    arm_obj = bpy.data.objects.new(name="arm_armature", object_data=arm_data)
    bpy.context.collection.objects.link(arm_obj)

    # Explicitly set armature to origin with identity transform
    arm_obj.location = (0, 0, 0)
    arm_obj.rotation_euler = (0, 0, 0)
    arm_obj.scale = (1, 1, 1)

    # Enter edit mode to create bones
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = arm_data.edit_bones

    # Bone 1: upper_arm (shoulder → elbow)
    upper = edit_bones.new("upper_arm")
    upper.head = SHOULDER
    upper.tail = ELBOW
    print(f"  upper_arm bone: {SHOULDER} -> {ELBOW}")

    # Bone 2: forearm (elbow → wrist), parented to upper_arm
    forearm = edit_bones.new("forearm")
    forearm.head = ELBOW
    forearm.tail = WRIST
    forearm.parent = upper
    print(f"  forearm bone: {ELBOW} -> {WRIST}")

    bpy.ops.object.mode_set(mode='OBJECT')

    return arm_obj


def flatten_mesh_hierarchy(meshes):
    """Clear parent transforms on all meshes (keep world position).
    This removes dependency on the Empty pivot nodes."""
    print(f"Flattening hierarchy for {len(meshes)} meshes...")
    count = 0
    for mesh_obj in meshes:
        if mesh_obj.parent:
            # Use matrix_world to preserve world position after unparenting
            matrix_world = mesh_obj.matrix_world.copy()
            mesh_obj.parent = None
            mesh_obj.matrix_world = matrix_world
            count += 1
    print(f"  Unparented {count} meshes from their parents")


def delete_empties(empties):
    """Remove Empty nodes — no longer needed after flattening mesh hierarchy."""
    print(f"Removing {len(empties)} empty nodes...")
    for empty_obj in empties:
        try:
            bpy.data.objects.remove(empty_obj, do_unlink=True)
        except Exception as e:
            print(f"  WARNING: Could not remove '{empty_obj.name}': {e}")
    print("  Done")


def parent_meshes_to_armature(meshes, arm_obj):
    """Parent all arm meshes to the armature with automatic weights."""
    print(f"Binding {len(meshes)} meshes to armature...")

    skipped = 0
    bound = 0
    for mesh_obj in meshes:
        # Select mesh object
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj

        try:
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')
            bound += 1
        except RuntimeError as e:
            print(f"  WARNING: Could not bind '{mesh_obj.name}': {e}")
            skipped += 1

    print(f"  Bound: {bound}, Skipped: {skipped}")


def export_glb(filepath):
    """Export scene as glTF 2.0 binary."""
    print(f"Exporting to {filepath}...")
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        export_apply=False,      # Preserve node transforms
        export_skins=True,
        export_morph=False,
        export_yup=False,         # Keep Z-up for consistency with existing model
        export_keep_originals=False,
    )
    print("Export complete.")


def main():
    print("=" * 60)
    print("Arm Skinning Tool — Adding Armature to Z-Anatomy Arm Model")
    print("=" * 60)

    clear_scene()
    meshes, empties = import_glb(INPUT_GLB)

    # Flatten mesh hierarchy BEFORE creating armature
    # This way meshes are at scene root with correct world positions
    flatten_mesh_hierarchy(meshes)

    # Remove Empties — they're no longer needed
    delete_empties(empties)

    # Create armature at origin with bones at correct world positions
    arm_obj = create_armature()

    # Bind meshes to armature
    parent_meshes_to_armature(meshes, arm_obj)

    export_glb(OUTPUT_GLB)

    print(f"\nDone! Output saved to: {OUTPUT_GLB}")


if __name__ == "__main__":
    main()
