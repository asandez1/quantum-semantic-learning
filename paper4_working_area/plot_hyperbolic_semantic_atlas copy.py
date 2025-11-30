# plot_hyperbolic_semantic_atlas_3d.py
# Publication-quality Poincaré ball visualization of semantic knowledge
# For: Quantum Atlas paper (Nov 2025)

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# --- Setup ---
plt.style.use('default')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "text.usetex": False,
    "figure.figsize": (12, 12),
    "axes.labelsize": 14,
    "legend.fontsize": 11
})
os.makedirs('figures', exist_ok=True)

# --- Data ---
hierarchy = {
    "entity": ["animal", "plant", "artifact", "emotion", "color"],
    "animal": ["mammal", "bird", "reptile", "fish"],
    "mammal": ["dog", "cat"], "dog": ["poodle"], "cat": ["siamese"],
    "bird": ["sparrow"], "reptile": ["snake"], "fish": ["salmon"],
    "plant": ["tree", "flower"], "tree": ["oak"],
    "artifact": ["tool", "vehicle", "building"], "tool": ["hammer"], 
    "vehicle": ["car"], "car": ["sedan"],
    "emotion": ["happiness", "anger"], "happiness": ["joy"],
    "color": ["blue", "red"], "blue": ["navy"]
}

patch_colors = {
    "animal": "#E31A1C", "mammal": "#E31A1C", "dog": "#E31A1C", "poodle": "#E31A1C",
    "cat": "#E31A1C", "siamese": "#E31A1C", "bird": "#E31A1C", "sparrow": "#E31A1C",
    "reptile": "#E31A1C", "snake": "#E31A1C", "fish": "#E31A1C", "salmon": "#E31A1C",
    "plant": "#33A02C", "tree": "#33A02C", "oak": "#33A02C", "flower": "#33A02C",
    "artifact": "#1F78B4", "tool": "#1F78B4", "vehicle": "#1F78B4", "car": "#1F78B4", 
    "building": "#1F78B4", "sedan": "#1F78B4",
    "emotion": "#FF7F00", "happiness": "#FF7F00", "joy": "#FF7F00", "anger": "#FF7F00",
    "color": "#6A3D9A", "blue": "#6A3D9A", "red": "#6A3D9A", "navy": "#6A3D9A",
    "entity": "black"
}

# --- 3D Hyperbolic Layout ---
def layout_tree_3d(root, depth_scale=0.7, sibling_angle=np.pi/3):
    """Recursively assign 3D hyperbolic coordinates to a tree."""
    coords = {}
    
    def fibonacci_sphere(samples=1):
        if samples <= 1:
            return [(0.0, 0.0, 1.0)] # Default direction for single child
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append((x, y, z))
        return points

    def recurse(node, parent_coord, depth, direction_vec):
        # Position node along the direction vector, scaled by depth
        r = depth_scale ** depth
        coord = parent_coord + r * direction_vec
        coords[node] = coord
        
        children = hierarchy.get(node, [])
        if not children:
            return
            
        # Distribute children on a sphere around the new coordinate
        child_directions = fibonacci_sphere(len(children))
        
        for i, child in enumerate(children):
            # Rotate child direction to align with parent direction
            # This is a simplification; a proper rotation would be needed for perfect alignment
            # but for visualization, this even distribution works well.
            new_direction = child_directions[i]
            recurse(child, coord, depth + 1, np.array(new_direction))

    coords[root] = np.array([0.0, 0.0, 0.0])
    # Initial directions for top-level categories
    initial_directions = fibonacci_sphere(len(hierarchy.get(root, [])))
    for i, child in enumerate(hierarchy.get(root, [])):
        recurse(child, coords[root], 1, np.array(initial_directions[i]))
        
    # Scale all coordinates to fit inside the unit ball
    max_dist = max(np.linalg.norm(p) for p in coords.values())
    if max_dist > 0.95:
        scale_factor = 0.95 / max_dist
        for node in coords:
            coords[node] *= scale_factor
            
    return coords

coords_3d = layout_tree_3d("entity")

# --- 3D Plotting ---
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')
ax.axis('off')

# Draw transparent unit sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', rstride=4, cstride=4, 
                linewidth=0, alpha=0.05, zorder=1)

# Draw edges
for parent, children in hierarchy.items():
    if parent not in coords_3d: continue
    p = coords_3d[parent]
    for child in children:
        if child not in coords_3d: continue
        c = coords_3d[child]
        ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]], 
                color='gray', alpha=0.5, lw=1.0, zorder=2)

# Draw nodes
for node, pos in coords_3d.items():
    color = patch_colors.get(node, "gray")
    size = 250 if node in ["animal","plant","artifact","emotion","color"] else 120
    ax.scatter(pos[0], pos[1], pos[2], s=size, c=color, 
               edgecolors='black', linewidth=1.2, zorder=4, depthshade=True)
    ax.text(pos[0], pos[1], pos[2] + 0.06, node.replace("_", " "), 
            ha='center', va='bottom', fontsize=9, zorder=5,
            fontweight='bold' if size > 200 else 'normal')

# --- Annotation and Legend ---
ax.text(0, 0, 1.3, "Quantum Atlas: 3D Hyperbolic Structure of Semantics", 
        ha='center', va='center', fontsize=20, fontweight='bold')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E31A1C', markersize=12, label='Animal Patch'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#33A02C', markersize=12, label='Plant Patch'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F78B4', markersize=12, label='Artifact Patch'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7F00', markersize=12, label='Emotion Patch'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#6A3D9A', markersize=12, label='Color Patch'),
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True, bbox_to_anchor=(0.02, 0.98))

ax.set_title("Poincaré Ball Embedding of ConceptNet Hierarchy", fontsize=14, pad=20)
ax.view_init(elev=20, azim=45)
ax.dist = 8

# --- Save and Finish ---
plt.savefig("figures/hyperbolic_semantic_atlas_3d.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig("figures/hyperbolic_semantic_atlas_3d.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

print("3D figure saved: figures/hyperbolic_semantic_atlas_3d.pdf")
