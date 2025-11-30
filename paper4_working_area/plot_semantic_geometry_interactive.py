#!/usr/bin/env python3
"""
Interactive Semantic Geometry Visualization for Paper 4:
"Quantum Advantage in Semantic Geometry: The Critical Role of Encoding and Entanglement"

Creates an interactive 3D Poincaré ball showing:
1. Semantic hierarchy as hyperbolic embedding
2. Animation showing "training" progression
3. Color-coded semantic patches (animals, plants, artifacts, etc.)

Output: figures/semantic_geometry_interactive.html
"""

import os
import math
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

os.makedirs('figures', exist_ok=True)
out_html = "figures/semantic_geometry_interactive.html"

# =============================================================================
# SEMANTIC HIERARCHY (ConceptNet-style)
# =============================================================================
hierarchy = {
    "entity": ["animal", "plant", "artifact", "emotion", "color"],
    "animal": ["mammal", "bird", "reptile", "fish"],
    "mammal": ["dog", "cat", "horse", "wolf"],
    "dog": ["poodle", "puppy"],
    "cat": ["siamese", "kitten"],
    "bird": ["sparrow", "eagle", "hawk"],
    "reptile": ["snake", "lizard"],
    "fish": ["salmon", "shark", "dolphin"],
    "plant": ["tree", "flower"],
    "tree": ["oak"],
    "flower": ["rose"],
    "artifact": ["tool", "vehicle", "building"],
    "tool": ["hammer", "screwdriver"],
    "vehicle": ["car", "sedan"],
    "building": ["house"],
    "emotion": ["happiness", "anger", "fear"],
    "happiness": ["joy"],
    "color": ["blue", "red", "green"],
    "blue": ["navy"],
}

# Color scheme by semantic domain
patch_colors = {
    # Animals - Red family
    "animal": "#E63946", "mammal": "#E63946", "dog": "#E63946", "poodle": "#E63946",
    "puppy": "#E63946", "cat": "#E63946", "siamese": "#E63946", "kitten": "#E63946",
    "horse": "#E63946", "wolf": "#E63946",
    "bird": "#E85D75", "sparrow": "#E85D75", "eagle": "#E85D75", "hawk": "#E85D75",
    "reptile": "#F07167", "snake": "#F07167", "lizard": "#F07167",
    "fish": "#F4A261", "salmon": "#F4A261", "shark": "#F4A261", "dolphin": "#F4A261",

    # Plants - Green family
    "plant": "#2D6A4F", "tree": "#40916C", "oak": "#52B788",
    "flower": "#74C69D", "rose": "#95D5B2",

    # Artifacts - Blue family
    "artifact": "#1D3557", "tool": "#457B9D", "hammer": "#457B9D", "screwdriver": "#457B9D",
    "vehicle": "#A8DADC", "car": "#A8DADC", "sedan": "#A8DADC",
    "building": "#F1FAEE", "house": "#F1FAEE",

    # Emotions - Orange family
    "emotion": "#E76F51", "happiness": "#F4A261", "joy": "#E9C46A",
    "anger": "#E76F51", "fear": "#264653",

    # Colors - Purple family
    "color": "#7B2CBF", "blue": "#5A189A", "navy": "#3C096C",
    "red": "#9D4EDD", "green": "#C77DFF",

    # Root
    "entity": "#FFFFFF",
}

# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================
def fibonacci_sphere(samples):
    """Generate evenly distributed points on a sphere."""
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
    """Convert hierarchy depth to Poincaré ball radius."""
    return 0.95 * (1 - math.exp(-base * depth))


def geodesic_arc(a, b, n_points=40, bulge=0.2):
    """Create a curved arc between two points (approximates geodesic)."""
    a = np.array(a)
    b = np.array(b)
    mid = 0.5 * (a + b)
    chord = b - a
    if np.allclose(chord, 0):
        return np.tile(a, (n_points, 1))
    arb = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(chord / (np.linalg.norm(chord) + 1e-9), arb)) > 0.95:
        arb = np.array([0.0, 1.0, 0.0])
    perp = np.cross(chord, arb)
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    ctrl = mid + perp * np.linalg.norm(chord) * bulge
    t = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)
    points = (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b
    return points


def layout_poincare(root="entity", depth_scale=0.65):
    """Layout hierarchy in Poincaré ball."""
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
        parent_dir = np.array(direction)
        parent_norm = np.linalg.norm(parent_dir)
        for i, child in enumerate(children):
            d = np.array(directions[i])
            if parent_norm > 1e-6:
                d = (0.35 * parent_dir / parent_norm + 0.65 * d)
                d = d / (np.linalg.norm(d) + 1e-9)
            recurse(child, depth + 1, d)

    coords[root] = np.array([0.0, 0.0, 0.0])
    levels[root] = 0
    top_children = hierarchy.get(root, [])
    top_dirs = fibonacci_sphere(len(top_children))
    for i, child in enumerate(top_children):
        recurse(child, 1, top_dirs[i])

    # Clamp to ball boundary
    for k, v in coords.items():
        n = np.linalg.norm(v)
        if n >= 0.96:
            coords[k] = v * (0.95 / n)

    return coords, levels


# =============================================================================
# BUILD PLOTLY TRACES
# =============================================================================
def build_traces(coords, levels, show_labels=True):
    """Build Plotly traces for nodes and edges."""
    # Node trace
    xs, ys, zs = [], [], []
    texts, sizes, colors = [], [], []
    major_nodes = {"entity", "animal", "plant", "artifact", "emotion", "color"}

    for node, pos in coords.items():
        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])
        depth = levels.get(node, 0)
        texts.append(f"{node}<br>Depth: {depth}")
        if node in major_nodes:
            sizes.append(14)
        elif depth <= 2:
            sizes.append(10)
        else:
            sizes.append(7)
        colors.append(patch_colors.get(node, "#888888"))

    node_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text' if show_labels else 'markers',
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color='#222222', width=1.5),
            opacity=0.95
        ),
        text=[t.split("<br>")[0] for t in texts] if show_labels else None,
        hovertext=texts,
        hoverinfo='text',
        textposition="top center",
        textfont=dict(size=9, color='white'),
        showlegend=False
    )

    # Edge trace (geodesic arcs)
    edge_x, edge_y, edge_z = [], [], []
    for parent, children in hierarchy.items():
        if parent not in coords:
            continue
        p = coords[parent]
        for child in children:
            if child not in coords:
                continue
            c = coords[child]
            depth_avg = (levels.get(parent, 0) + levels.get(child, 0)) / 2.0
            pts = geodesic_arc(p, c, n_points=30, bulge=0.15 + 0.05 * depth_avg)
            for pt in pts:
                edge_x.append(pt[0])
                edge_y.append(pt[1])
                edge_z.append(pt[2])
            edge_x.append(None)
            edge_y.append(None)
            edge_z.append(None)

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(180,180,210,0.5)', width=3),
        hoverinfo='none',
        showlegend=False
    )

    return [edge_trace, node_trace]


# =============================================================================
# CREATE BOUNDARY SPHERE
# =============================================================================
def create_boundary_sphere(opacity=0.08):
    """Create a translucent boundary sphere."""
    phi = np.linspace(0, 2 * np.pi, 50)
    theta = np.linspace(0, np.pi, 25)
    phi, theta = np.meshgrid(phi, theta)
    xs = (np.cos(phi) * np.sin(theta)).flatten()
    ys = (np.sin(phi) * np.sin(theta)).flatten()
    zs = (np.cos(theta)).flatten()

    sphere = go.Mesh3d(
        x=xs, y=ys, z=zs,
        opacity=opacity,
        color='rgb(100,105,115)',
        alphahull=0,
        hoverinfo='none',
        showscale=False
    )
    return sphere


# =============================================================================
# CREATE ANIMATION FRAMES (Simulating Training)
# =============================================================================
def create_frames():
    """Create animation frames showing geometry refinement."""
    # Vary the depth scale to simulate training progression
    scales = np.linspace(0.3, 1.0, 15)
    frames = []

    for i, scale in enumerate(scales):
        coords, levels = layout_poincare("entity", depth_scale=float(scale))
        traces = build_traces(coords, levels, show_labels=(i == len(scales) - 1))
        frame_name = f"step_{i}"
        frames.append(go.Frame(data=traces, name=frame_name))

    return frames, scales


# =============================================================================
# BUILD FIGURE
# =============================================================================
def build_figure():
    """Build the complete interactive figure."""
    # Initial state
    initial_coords, initial_levels = layout_poincare("entity", depth_scale=0.3)
    initial_traces = build_traces(initial_coords, initial_levels, show_labels=True)

    # Boundary sphere
    sphere = create_boundary_sphere(opacity=0.1)

    # Animation frames
    frames, scales = create_frames()

    # Slider steps
    slider_steps = [
        {
            "args": [[f"step_{i}"], {"frame": {"duration": 200, "redraw": True}, "mode": "immediate"}],
            "label": f"{s:.2f}",
            "method": "animate"
        }
        for i, s in enumerate(scales)
    ]

    # Layout
    layout = go.Layout(
        paper_bgcolor='rgb(15,17,22)',
        plot_bgcolor='rgb(15,17,22)',
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            bgcolor='rgb(10,12,18)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.3, z=1.0))
        ),
        margin=dict(t=100, l=0, r=0, b=60),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.0,
                x=1.1,
                xanchor="right",
                buttons=[
                    dict(
                        label="▶ Train",
                        method="animate",
                        args=[None, {"frame": {"duration": 200, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 150}}]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate"}]
                    )
                ]
            )
        ],
        sliders=[{
            "pad": {"b": 10, "t": 40},
            "len": 0.85,
            "x": 0.075,
            "y": 0.02,
            "currentvalue": {
                "prefix": "Training Progress: ",
                "font": {"color": "white", "size": 12}
            },
            "steps": slider_steps,
            "tickcolor": "white",
            "font": {"color": "white"}
        }],
        annotations=[
            dict(
                text="<b>Quantum Semantic Geometry</b>",
                font=dict(size=22, color='white'),
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.97
            ),
            dict(
                text="Poincaré Ball Embedding of ConceptNet Hierarchy",
                font=dict(size=13, color='#AAAAAA'),
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.93
            ),
            dict(
                text="🔴 Animals  🟢 Plants  🔵 Artifacts  🟠 Emotions  🟣 Colors",
                font=dict(size=11, color='#888888'),
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.89
            ),
            dict(
                text="Drag to rotate • Scroll to zoom • Slider simulates training",
                font=dict(size=10, color='#666666'),
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.01
            ),
        ]
    )

    # Assemble figure
    fig = go.Figure(
        data=[sphere] + initial_traces,
        layout=layout,
        frames=frames
    )

    return fig


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Generating interactive semantic geometry visualization...")

    fig = build_figure()
    plot(fig, filename=out_html, auto_open=False, include_plotlyjs='cdn')

    print(f"Saved: {out_html}")
    print("\nOpen in browser to interact with the 3D visualization.")
    print("Features:")
    print("  - Drag to rotate the Poincaré ball")
    print("  - Scroll to zoom")
    print("  - Use slider to simulate training progression")
    print("  - Hover over nodes to see concept details")
