# figures/hyperbolic_semantic_atlas_interactive.py
# Produces an interactive HTML (Plotly + WebGL) for the Poincaré semantic atlas.
# Provenance: original file path: /mnt/data/plot_hyperbolic_semantic_atlas.py

import os
import math
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

# --- provenance (original file) ---
original_file = "/mnt/data/plot_hyperbolic_semantic_atlas.py"
print("Original script (provenance):", original_file)

os.makedirs('figures', exist_ok=True)
out_html = "figures/hyperbolic_semantic_atlas_interactive.html"

# --- Hierarchy & palette (same semantic graph) ---
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
    "animal": "#A62828", "mammal": "#A62828", "dog": "#A62828", "poodle": "#A62828",
    "cat": "#A62828", "siamese": "#A62828", "bird": "#A62828", "sparrow": "#A62828",
    "reptile": "#A62828", "snake": "#A62828", "fish": "#A62828", "salmon": "#A62828",
    "plant": "#2F7D3A", "tree": "#2F7D3A", "oak": "#2F7D3A", "flower": "#2F7D3A",
    "artifact": "#2C5F85", "tool": "#2C5F85", "vehicle": "#2C5F85", "car": "#2C5F85",
    "building": "#2C5F85", "sedan": "#2C5F85",
    "emotion": "#B46F2F", "happiness": "#B46F2F", "joy": "#B46F2F", "anger": "#B46F2F",
    "color": "#6B4C7A", "blue": "#6B4C7A", "red": "#6B4C7A", "navy": "#6B4C7A",
    "entity": "#DDDDDD"
}

# --- geometry utilities (same logic as the improved static script) ---
def fibonacci_sphere(samples):
    if samples == 1:
        return [(0.0, 0.0, 1.0)]
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points

def hyperbolic_radius(depth, base=0.65):
    return 0.95 * (1 - math.exp(-base * depth))

def great_circle_arc(a, b, n_points=60, bulge=0.25):
    a = np.array(a); b = np.array(b)
    mid = 0.5 * (a + b)
    chord = b - a
    if np.allclose(chord, 0):
        return np.tile(a, (n_points, 1))
    arb = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(chord / (np.linalg.norm(chord)+1e-9), arb)) > 0.95:
        arb = np.array([0.0, 1.0, 0.0])
    perp = np.cross(chord, arb)
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    ctrl = mid + perp * np.linalg.norm(chord) * bulge
    t = np.linspace(0.0, 1.0, n_points).reshape(-1,1)
    points = (1 - t)**2 * a + 2 * (1 - t) * t * ctrl + t**2 * b
    return points

def layout_poincare(root="entity", depth_scale=0.65):
    coords = {}
    levels = {}
    def recurse(node, depth, direction):
        r = hyperbolic_radius(depth, base=depth_scale)
        coords[node] = np.array(direction) * r
        levels[node] = depth
        children = hierarchy.get(node, [])
        if not children: return
        directions = fibonacci_sphere(len(children))
        parent_dir = np.array(direction)
        parent_norm = np.linalg.norm(parent_dir)
        for i, child in enumerate(children):
            d = np.array(directions[i])
            if parent_norm > 1e-6:
                d = (0.35 * parent_dir / parent_norm + 0.65 * d)
                d = d / (np.linalg.norm(d) + 1e-9)
            recurse(child, depth + 1, d)
    coords[root] = np.array([0.0,0.0,0.0]); levels[root] = 0
    top_children = hierarchy.get(root, [])
    top_dirs = fibonacci_sphere(len(top_children))
    for i, child in enumerate(top_children):
        recurse(child, 1, top_dirs[i])
    for k,v in coords.items():
        n = np.linalg.norm(v)
        if n >= 0.96:
            coords[k] = v * (0.95/n)
    return coords, levels

# --- Helpers to build Plotly traces for a given layout ---
def build_traces(coords, levels):
    # nodes
    xs=[]; ys=[]; zs=[]; texts=[]; sizes=[]; colors=[]
    large_nodes = {"animal", "plant", "artifact", "emotion", "color", "entity"}
    for node, pos in coords.items():
        xs.append(pos[0]); ys.append(pos[1]); zs.append(pos[2])
        depth = levels.get(node,0)
        texts.append(f"{node} (depth {depth})")
        if node in large_nodes:
            sizes.append(12)
            colors.append(patch_colors.get(node, "#999999"))
        else:
            sizes.append(6)
            colors.append(patch_colors.get(node, "#888888"))

    node_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, line=dict(color='#111111', width=1.0)),
        text=[t.split(" ")[0] for t in texts],  # short label
        hovertext=texts,
        hoverinfo='text',
        textposition="top center",
        showlegend=False
    )

    # edges combined into single trace with None separators (for performance)
    edge_x = []; edge_y = []; edge_z = []; edge_color = []
    for parent, children in hierarchy.items():
        if parent not in coords: continue
        p = coords[parent]
        for child in children:
            if child not in coords: continue
            c = coords[child]
            pts = great_circle_arc(p, c, n_points=40, bulge=0.18 + 0.06 * ((levels.get(parent,0)+levels.get(child,0))/2.0))
            # append segment coordinates with None separators
            for pt in pts:
                edge_x.append(pt[0]); edge_y.append(pt[1]); edge_z.append(pt[2])
            edge_x.append(None); edge_y.append(None); edge_z.append(None)

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(200,200,230,0.55)', width=5),
        hoverinfo='none',
        showlegend=False
    )

    return [edge_trace, node_trace]

# --- Create frames for interactive slider: vary base parameter (simulates training radial change) ---
bases = np.linspace(0.35, 1.2, 12)  # slider positions
frames = []
initial_coords, initial_levels = layout_poincare("entity", depth_scale=bases[0])
initial_data = build_traces(initial_coords, initial_levels)

for b in bases:
    coords_b, levels_b = layout_poincare("entity", depth_scale=float(b))
    traces_b = build_traces(coords_b, levels_b)
    # Each frame's 'data' must match the top-level data trace order & count
    frames.append(go.Frame(data=traces_b, name=f"{b:.3f}"))

# --- Build the static boundary sphere (thin mesh) for visual reference ---
# Use a light wireframe sphere plotted as mesh3d (subtle)
phi = np.linspace(0, 2*np.pi, 60)
theta = np.linspace(0, np.pi, 30)
phi, theta = np.meshgrid(phi, theta)
xs = (np.cos(phi) * np.sin(theta)).flatten()
ys = (np.sin(phi) * np.sin(theta)).flatten()
zs = (np.cos(theta)).flatten()
sphere = go.Mesh3d(
    x=xs, y=ys, z=zs,
    opacity=0.12,
    color='rgb(90,95,100)',
    alphahull=0,
    hoverinfo='none',
    showscale=False
)

# --- Layout: dark theme and remote title above globe ---
layout = go.Layout(
    paper_bgcolor='rgb(8,10,12)',
    plot_bgcolor='rgb(8,10,12)',
    title=dict(text="", x=0.5),
    scene=dict(
        xaxis=dict(showbackground=False, visible=False, showticklabels=False),
        yaxis=dict(showbackground=False, visible=False, showticklabels=False),
        zaxis=dict(showbackground=False, visible=False, showticklabels=False),
        bgcolor='rgb(0,0,0)',
        aspectmode='data',
        camera=dict(eye=dict(x=1.4, y=1.2, z=0.9))
    ),
    margin=dict(t=120, l=0, r=0, b=0),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            y=1.02,
            x=1.12,
            xanchor="right",
            yanchor="top",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 250, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 200}}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate", "transition": {"duration": 0}}])
            ]
        )
    ],
    sliders=[{
        "pad": {"b": 10, "t": 60},
        "len": 0.8,
        "x": 0.1,
        "y": 0.02,
        "currentvalue": {"prefix": "Training scale: ", "font": {"color": "white"}},
        "steps": [
            {
                "args": [[f"{b:.3f}"], {"frame": {"duration": 250, "redraw": True}, "mode": "immediate"}],
                "label": f"{b:.2f}",
                "method": "animate"
            } for b in bases
        ]
    }],
    annotations=[
        dict(
            text="<b>Quantum Atlas — Interactive Poincaré Ball</b>",
            font=dict(size=20, color='white'),
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.98
        ),
        dict(
            text="Drag to rotate • Scroll to zoom • Hover nodes for details",
            font=dict(size=11, color='lightgray'),
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.945
        ),
        dict(
            text=f"Provenance: {original_file}",
            font=dict(size=9, color='gray'),
            showarrow=False,
            xref="paper", yref="paper",
            x=0.01, y=0.02, align="left"
        )
    ]
)

# --- Assemble the figure with initial data + sphere + frames ---
fig = go.Figure(
    data=[sphere] + initial_data,
    layout=layout,
    frames=frames
)

# Save as single-file HTML (self-contained)
plot(fig, filename=out_html, auto_open=False, include_plotlyjs='cdn')
print("Interactive HTML saved to:", out_html)
