# hyperbolic_semantic_atlas_improved.py
# Improved Poincaré ball visualization (geometric + entanglement threads + paper-ready)
# Based on user's original file: /mnt/data/plot_hyperbolic_semantic_atlas.py

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib import colors as mcolors

# --- provenance (original file) ---
original_file = "/mnt/data/plot_hyperbolic_semantic_atlas.py"
print("Original script (provenance):", original_file)

# --- Output folder ---
os.makedirs('figures', exist_ok=True)

# --- Data (same hierarchy, slightly cleaned) ---
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

# Professional, muted, geometric palette (no neon)
patch_colors = {
    "animal": "#A62828", "mammal": "#A62828", "dog": "#A62828", "poodle": "#A62828",
    "cat": "#A62828", "siamese": "#A62828", "bird": "#A62828", "sparrow": "#A62828",
    "reptile": "#A62828", "snake": "#A62828", "fish": "#A62828", "salmon": "#A62828",
    "plant": "#2F7D3A", "tree": "#2F7D3A", "oak": "#2F7D3A", "flower": "#2F7D3A",
    "artifact": "#2C5F85", "tool": "#2C5F85", "vehicle": "#2C5F85", "car": "#2C5F85",
    "building": "#2C5F85", "sedan": "#2C5F85",
    "emotion": "#B46F2F", "happiness": "#B46F2F", "joy": "#B46F2F", "anger": "#B46F2F",
    "color": "#6B4C7A", "blue": "#6B4C7A", "red": "#6B4C7A", "navy": "#6B4C7A",
    "entity": "#222222"
}

# --- Utilities ---
def fibonacci_sphere(samples):
    """Return well-distributed points on unit sphere (x,y,z)."""
    points = []
    if samples == 1:
        return [(0.0, 0.0, 1.0)]
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y from 1 to -1
        radius = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points

def hyperbolic_radius(depth, base=0.65):
    """Map tree depth to radius in Poincaré ball (0..1) using a tanh-like shrinkage.
       base ~ controls spacing between layers. Deeper nodes approach the boundary quickly.
    """
    # Depth 0 => 0.0, depth increases -> radius approaches 0.95
    # Use scaled logistic/tanh mapping for smoothness
    return 0.95 * (1 - math.exp(-base * depth))

def great_circle_arc(a, b, n_points=60, bulge=0.25):
    """Create a smooth curved arc between 3D points a and b inside unit ball.
       We approximate an 'entanglement thread' by constructing a quadratic Bezier-like curve
       that bows outward from the chord to suggest curvature.
       bulge controls how much arching occurs (positive -> outward).
    """
    a = np.array(a)
    b = np.array(b)
    mid = 0.5 * (a + b)
    # outward direction: perpendicular to chord; use cross product with arbitrary vector
    chord = b - a
    # choose a stable perpendicular basis vector
    if np.allclose(chord, 0):
        return np.tile(a, (n_points, 1))
    # choose arbitrary vector not parallel to chord
    arb = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(chord / np.linalg.norm(chord), arb)) > 0.95:
        arb = np.array([0.0, 1.0, 0.0])
    perp = np.cross(chord, arb)
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    # control point placed outward proportional to chord length and bulge value
    ctrl = mid + perp * np.linalg.norm(chord) * bulge
    # Quadratic Bezier sampling
    t = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)
    points = (1 - t)**2 * a + 2 * (1 - t) * t * ctrl + t**2 * b
    return points

# --- Layout: Poincaré-style placement using directions + hyperbolic radial mapping ---
def layout_poincare(root="entity", depth_scale=0.65):
    coords = {}
    levels = {}

    def recurse(node, depth, direction):
        r = hyperbolic_radius(depth, base=depth_scale)
        coords[node] = np.array(direction) * r
        levels[node] = depth
        children = hierarchy.get(node, [])
        if not children:
            return
        directions = fibonacci_sphere(len(children))
        # rotate/align directions roughly with parent direction for coherent branching
        # simple alignment: if parent direction is near zero (root), keep direct directions
        parent_dir = np.array(direction)
        parent_norm = np.linalg.norm(parent_dir)
        for i, child in enumerate(children):
            d = np.array(directions[i])
            if parent_norm > 1e-6:
                # blend the child direction toward parent direction so subtree points outward from center
                d = (0.35 * parent_dir / parent_norm + 0.65 * d)
                d = d / np.linalg.norm(d)
            recurse(child, depth + 1, d)

    coords[root] = np.array([0.0, 0.0, 0.0])
    levels[root] = 0
    top_children = hierarchy.get(root, [])
    top_dirs = fibonacci_sphere(len(top_children))
    for i, child in enumerate(top_children):
        recurse(child, 1, top_dirs[i])

    # Safety scaling: clamp to < 0.95
    for k, v in coords.items():
        norm = np.linalg.norm(v)
        if norm >= 0.96:
            coords[k] = v * (0.95 / norm)
    return coords, levels

coords, levels = layout_poincare("entity", depth_scale=0.75)

# --- Figure setup (paper-ready: subtle off-white background with polished layout) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.figsize": (10, 10),
    "axes.titlesize": 14,
    "axes.labelsize": 11
})

fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor((0.98, 0.98, 0.98))  # very light paper gray
ax.patch.set_alpha(1.0)
ax.grid(False)
ax.axis('off')

# unit sphere surface (Poincaré ball boundary) - subtle shading for depth
u = np.linspace(0, 2 * np.pi, 120)
v = np.linspace(0, np.pi, 80)
x_s = np.outer(np.cos(u), np.sin(v))
y_s = np.outer(np.sin(u), np.sin(v))
z_s = np.outer(np.ones_like(u), np.cos(v))
# light-to-shadow color map for the sphere
light = np.array([0.92, 0.93, 0.95])
shadow = np.array([0.82, 0.83, 0.86])
# compute simple shading by z coordinate
shade = 0.5 * (z_s - z_s.min()) / (z_s.max() - z_s.min()) + 0.0
surf_colors = light[None, :] * (1 - shade[..., None]) + shadow[None, :] * (shade[..., None])
ax.plot_surface(x_s, y_s, z_s, rstride=4, cstride=4, facecolors=surf_colors,
                linewidth=0, alpha=0.40, zorder=0, antialiased=False)

# draw subtle equatorial guide lines (thin, faint)
theta_vals = np.linspace(0, 2 * np.pi, 80)
for elev in [0.0, 0.35]:
    xg = np.cos(theta_vals) * np.cos(elev)
    yg = np.sin(theta_vals) * np.cos(elev)
    zg = np.ones_like(theta_vals) * math.sin(elev)
    ax.plot(xg * 0.995, yg * 0.995, zg * 0.995, color=(0.7,0.7,0.7), lw=0.4, alpha=0.22, zorder=0)

# --- Draw entanglement threads (curved edges) ---
for parent, children in hierarchy.items():
    if parent not in coords:
        continue
    p = coords[parent]
    for child in children:
        if child not in coords:
            continue
        c = coords[child]
        # produce curved arc with bulge depending on levels (so deeper links are slightly more arched)
        depth_mean = (levels.get(parent, 0) + levels.get(child, 0)) / 2.0
        bulge = 0.18 + 0.06 * depth_mean  # mild variation
        pts = great_circle_arc(p, c, n_points=120, bulge=bulge)
        # use a muted gradient for thread color based on patch membership (blend gray + patch)
        parent_color = mcolors.to_rgb(patch_colors.get(parent, "#7f7f7f"))
        thread_color = tuple(0.35 * np.array(parent_color) + 0.65 * np.array([0.4,0.42,0.45]))
        # vary alpha along the thread (faint ends, slightly stronger mid)
        alphas = np.linspace(0.08, 0.35, len(pts))
        for i in range(len(pts)-1):
            seg = np.stack([pts[i], pts[i+1]])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    color=thread_color,
                    alpha=float(alphas[i]),
                    lw=0.9,
                    solid_capstyle='round',
                    zorder=1)

# --- Draw nodes (geometric spheres approximation via scatter) ---
# We'll render nodes in two passes: larger root/patch nodes, then small leaves for crisp layering
large_nodes = {"animal", "plant", "artifact", "emotion", "color", "entity"}
for node, pos in coords.items():
    pos = np.array(pos)
    if node in large_nodes:
        size = 260
        ec = (0.12, 0.12, 0.12)
        fc = patch_colors.get(node, "#777777")
        ax.scatter(pos[0], pos[1], pos[2], s=size, color=fc, edgecolors=ec, linewidths=1.4,
                   zorder=4)
# small nodes
for node, pos in coords.items():
    if node in large_nodes:
        continue
    pos = np.array(pos)
    size = 70
    fc = patch_colors.get(node, "#888888")
    ax.scatter(pos[0], pos[1], pos[2], s=size, color=fc, edgecolors=(0.15,0.15,0.15), linewidths=0.9,
               zorder=5)

# --- Node labels (subtle, professional) ---
for node, pos in coords.items():
    pos = np.array(pos)
    depth = levels.get(node, 0)
    # label offset: push labels slightly outward from node to improve readability
    if np.linalg.norm(pos) < 1e-6:
        offset_dir = np.array([0.0, -0.08, 0.0])
    else:
        offset_dir = pos / (np.linalg.norm(pos) + 1e-9) * 0.06
    # font weight for patches
    weight = 'semibold' if node in large_nodes else 'normal'
    fontsize = 10 if node in large_nodes else 8.5
    ax.text(pos[0] + offset_dir[0], pos[1] + offset_dir[1], pos[2] + offset_dir[2],
            node.replace("_", " "), horizontalalignment='center', verticalalignment='center',
            fontsize=fontsize, zorder=6, fontweight=weight, color=(0.08,0.08,0.08))

# --- Title, subtitle with training explanation hint (paper style) ---
ax.text(0, 0, 1.12, "Quantum Atlas — Geometric Poincaré Representation of Semantics",
        ha='center', va='center', fontsize=16, fontweight='bold', zorder=10)
ax.text(0, 0, 1.06,
        "Entanglement threads show semantic links; radial positions encode hierarchical depth (Poincaré-style).",
        ha='center', va='center', fontsize=9.5, zorder=10, alpha=0.9)

# --- Legend (custom, compact, paper-friendly) ---
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=patch_colors['animal'], markersize=9, label='Animal patch'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=patch_colors['plant'], markersize=9, label='Plant patch'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=patch_colors['artifact'], markersize=9, label='Artifact patch'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=patch_colors['emotion'], markersize=9, label='Emotion patch'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=patch_colors['color'], markersize=9, label='Color patch')
]
leg = ax.legend(handles=legend_elements, loc='upper left', frameon=True,
                bbox_to_anchor=(-0.02, 1.02), fontsize=9, handlelength=1.0)
leg.get_frame().set_edgecolor((0.85,0.85,0.85))
leg.get_frame().set_facecolor((0.99,0.99,0.99))
leg.get_frame().set_alpha(0.92)

# --- Viewing angle and final polish ---
ax.view_init(elev=18, azim=36)
# Slightly increase perspective effect
try:
    ax.dist = 7.8
except Exception:
    pass

# Tight layout and save high-resolution files (paper-ready)
plt.tight_layout()
png_path = "figures/hyperbolic_semantic_atlas_improved.png"
pdf_path = "figures/hyperbolic_semantic_atlas_improved.pdf"
plt.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.04)
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.04)
print("Saved:", png_path, pdf_path)

# If running in interactive environment, show the figure:
# plt.show()
