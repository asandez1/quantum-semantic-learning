#!/usr/bin/env python3
"""
Publication Figures for Paper 4:
"Quantum Advantage in Semantic Geometry: The Critical Role of Encoding and Entanglement"

Generates high-quality figures for the manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import os

# Create figures directory
os.makedirs('figures', exist_ok=True)

# =============================================================================
# COLOR SCHEME (Publication-ready, colorblind-friendly)
# =============================================================================
COLORS = {
    'quantum_blue': '#2E86AB',
    'success_green': '#28A745',
    'warning_orange': '#F18F01',
    'error_red': '#C73E1D',
    'neutral_gray': '#6C757D',
    'highlight_purple': '#7B2CBF',
    'background': '#FAFAFA',
    'dark_text': '#212529',
    'light_text': '#6C757D',

    # Semantic patches
    'animal': '#A62828',
    'plant': '#2F7D3A',
    'artifact': '#2C5F85',
    'emotion': '#B46F2F',
    'color': '#6B4C7A',
    'entity': '#555555',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


# =============================================================================
# FIGURE 1: Encoding Hierarchy (132× Gap)
# =============================================================================
def plot_encoding_hierarchy():
    """Bar chart showing the 132× encoding gap."""
    fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size

    encodings = ['DIRECT', 'CONCAT', 'DIFFERENCE']
    correlations = [0.9894, 0.5861, 0.0075]
    colors = [COLORS['success_green'], COLORS['warning_orange'], COLORS['error_red']]

    bars = ax.bar(encodings, correlations, color=colors, edgecolor='black', linewidth=1.2, width=0.55)

    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.annotate(f'r = {corr:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    # Add the 132× annotation
    ax.annotate('', xy=(0, 0.95), xytext=(2, 0.05),
                arrowprops=dict(arrowstyle='<->', color=COLORS['highlight_purple'], lw=2))
    ax.text(1, 0.5, '132×\nGap', ha='center', va='center',
            fontsize=13, fontweight='bold', color=COLORS['highlight_purple'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['highlight_purple']))

    ax.set_ylabel('Correlation (r)', fontsize=11)
    ax.set_xlabel('Encoding Strategy', fontsize=11)
    ax.set_title('Encoding Hierarchy: 132× Performance Gap\n(IBM ibm_fez, 156 qubits)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.9, color=COLORS['neutral_gray'], linestyle='--', alpha=0.5, label='Target (0.9)')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/fig1_encoding_hierarchy.png', facecolor='white')
    plt.savefig('figures/fig1_encoding_hierarchy.pdf', facecolor='white')
    print("Saved: figures/fig1_encoding_hierarchy.png/pdf")
    plt.close()


# =============================================================================
# FIGURE 2: Quantum Advantage Evidence
# =============================================================================
def plot_quantum_advantage():
    """Triple panel showing entanglement, hardware, and superposition advantages."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Entanglement Advantage (+0.99)
    ax1 = axes[0]
    circuits = ['Entangled\n(CX gates)', 'Product\n(No CX)']
    corrs = [0.68, -0.31]
    colors = [COLORS['quantum_blue'], COLORS['error_red']]
    bars1 = ax1.bar(circuits, corrs, color=colors, edgecolor='black', linewidth=1.2, width=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.set_ylabel('Correlation (r)')
    ax1.set_title('A) Entanglement Advantage', fontweight='bold')
    ax1.set_ylim(-0.5, 0.9)

    # Annotation for +0.99 effect
    ax1.annotate('', xy=(0, 0.68), xytext=(1, -0.31),
                arrowprops=dict(arrowstyle='<->', color=COLORS['highlight_purple'], lw=2.5))
    ax1.text(0.5, 0.2, '+0.99', ha='center', va='center',
            fontsize=16, fontweight='bold', color=COLORS['highlight_purple'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['highlight_purple'], lw=2))

    for bar, corr in zip(bars1, corrs):
        height = bar.get_height()
        offset = 5 if height > 0 else -15
        ax1.annotate(f'{corr:+.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

    # Panel B: Hardware Advantage (93% better)
    ax2 = axes[1]
    platforms = ['Hardware\n(ibm_fez)', 'Simulator\n(Qiskit Aer)']
    losses = [0.018, 0.211]
    colors2 = [COLORS['success_green'], COLORS['neutral_gray']]
    bars2 = ax2.bar(platforms, losses, color=colors2, edgecolor='black', linewidth=1.2, width=0.5)
    ax2.set_ylabel('Training Loss')
    ax2.set_title('B) Hardware Advantage', fontweight='bold')
    ax2.set_ylim(0, 0.3)

    # Annotation for 93% better
    ax2.annotate('', xy=(0, 0.018), xytext=(1, 0.211),
                arrowprops=dict(arrowstyle='<->', color=COLORS['highlight_purple'], lw=2.5))
    ax2.text(0.5, 0.12, '93%\nbetter', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['highlight_purple'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['highlight_purple'], lw=2))

    for bar, loss in zip(bars2, losses):
        ax2.annotate(f'{loss:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontweight='bold')

    # Panel C: Learning Effect (+1.31)
    ax3 = axes[2]
    states = ['Random\nParameters', 'Trained\nParameters']
    corrs3 = [-0.62, 0.68]
    colors3 = [COLORS['neutral_gray'], COLORS['success_green']]
    bars3 = ax3.bar(states, corrs3, color=colors3, edgecolor='black', linewidth=1.2, width=0.5)
    ax3.axhline(y=0, color='black', linewidth=0.8)
    ax3.set_ylabel('Correlation (r)')
    ax3.set_title('C) Superposition & Learning', fontweight='bold')
    ax3.set_ylim(-0.8, 0.9)

    # Annotation for +1.31 effect
    ax3.annotate('', xy=(1, 0.68), xytext=(0, -0.62),
                arrowprops=dict(arrowstyle='->', color=COLORS['highlight_purple'], lw=3,
                               connectionstyle='arc3,rad=0.3'))
    ax3.text(0.5, 0.0, '+1.31', ha='center', va='center',
            fontsize=16, fontweight='bold', color=COLORS['highlight_purple'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['highlight_purple'], lw=2))

    for bar, corr in zip(bars3, corrs3):
        height = bar.get_height()
        offset = 5 if height > 0 else -15
        ax3.annotate(f'{corr:+.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

    fig.suptitle('Quantum Advantage: Three Sources of Measurable Improvement',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('figures/fig2_quantum_advantage.png', facecolor='white')
    plt.savefig('figures/fig2_quantum_advantage.pdf', facecolor='white')
    print("Saved: figures/fig2_quantum_advantage.png/pdf")
    plt.close()


# =============================================================================
# FIGURE 3: Seven-Version Ablation Study
# =============================================================================
def plot_ablation_study():
    """Horizontal bar chart showing training effects for V1-V7."""
    fig, ax = plt.subplots(figsize=(8, 4.5))  # Reduced size

    versions = ['V7: Dense+CNOT', 'V6: Dense+CRz', 'V5: Global Parity',
                'V4: Dense+Ancilla', 'V3: Sparse+Ancilla', 'V2: Separate CX', 'V1: Interference']
    effects = [0.10, -0.08, -0.50, 0.46, 1.22, -0.02, -0.04]

    # Color by success/failure
    colors = []
    for e in effects:
        if e > 0.5:
            colors.append(COLORS['success_green'])
        elif e > 0:
            colors.append(COLORS['warning_orange'])
        else:
            colors.append(COLORS['error_red'])

    y_pos = np.arange(len(versions))
    bars = ax.barh(y_pos, effects, color=colors, edgecolor='black', linewidth=1.2, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(versions)
    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.set_xlabel('Training Effect (Trained - Random Correlation)', fontsize=12)
    ax.set_title('Seven-Version Ablation Study: Which Architecture Can Learn?',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-0.7, 1.4)

    # Add value labels
    for bar, effect in zip(bars, effects):
        width = bar.get_width()
        offset = 5 if width > 0 else -5
        ha = 'left' if width > 0 else 'right'
        ax.annotate(f'{effect:+.2f}', xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(offset, 0), textcoords="offset points",
                    ha=ha, va='center', fontweight='bold', fontsize=11)

    # Highlight V3 as winner
    ax.annotate('WINNER', xy=(1.22, 4), xytext=(1.0, 5.5),
                fontsize=12, fontweight='bold', color=COLORS['success_green'],
                arrowprops=dict(arrowstyle='->', color=COLORS['success_green'], lw=2))

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['success_green'], edgecolor='black', label='Strong learning (>0.5)'),
        mpatches.Patch(facecolor=COLORS['warning_orange'], edgecolor='black', label='Moderate (0-0.5)'),
        mpatches.Patch(facecolor=COLORS['error_red'], edgecolor='black', label='No learning (<0)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/fig3_ablation_study.png', facecolor='white')
    plt.savefig('figures/fig3_ablation_study.pdf', facecolor='white')
    print("Saved: figures/fig3_ablation_study.png/pdf")
    plt.close()


# =============================================================================
# FIGURE 4: V3 Circuit Architecture
# =============================================================================
def plot_v3_architecture():
    """Schematic diagram of the V3 (Sparse+Ancilla) winning architecture."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, 'V3 Architecture: Sparse Encoding + Ancilla Measurement',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(6, 7.0, 'The Winning Configuration (+1.22 Training Effect)',
            ha='center', va='center', fontsize=11, style='italic', color=COLORS['light_text'])

    # Qubit lines
    qubit_y = [6.0, 5.5, 5.0, 4.0, 3.5, 3.0, 1.5]
    qubit_labels = ['q₀ (v1[0])', 'q₁ (v1[1])', 'q₂ (v1[2])',
                    'q₃ (v2[0])', 'q₄ (v2[1])', 'q₅ (v2[2])',
                    'q₆ (Ancilla)']

    for i, (y, label) in enumerate(zip(qubit_y, qubit_labels)):
        # Qubit line
        color = COLORS['quantum_blue'] if i < 3 else (COLORS['warning_orange'] if i < 6 else COLORS['highlight_purple'])
        ax.plot([1, 11], [y, y], color=color, linewidth=2, alpha=0.7)
        # Label
        ax.text(0.5, y, label, ha='right', va='center', fontsize=9,
                color=color, fontweight='bold')
        # Initial state
        ax.text(1.2, y, '|0⟩', ha='center', va='center', fontsize=9)

    # Stage labels
    stages = [
        (2.0, 'Encoding\n(RY)'),
        (4.0, 'Layer 1\n(RY)'),
        (6.0, 'Entangle\n(CX)'),
        (8.0, 'Layer 2\n(RY)'),
        (10.0, 'Measure'),
    ]
    for x, label in stages:
        ax.text(x, 0.7, label, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

    # Encoding gates (RY) for v1
    for i, y in enumerate(qubit_y[:3]):
        rect = FancyBboxPatch((1.7, y-0.2), 0.6, 0.4, boxstyle="round,pad=0.02",
                              facecolor=COLORS['quantum_blue'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(2.0, y, 'RY', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    # Encoding gates (RY) for v2
    for i, y in enumerate(qubit_y[3:6]):
        rect = FancyBboxPatch((1.7, y-0.2), 0.6, 0.4, boxstyle="round,pad=0.02",
                              facecolor=COLORS['warning_orange'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(2.0, y, 'RY', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    # Trainable Layer 1 (all qubits)
    for y in qubit_y:
        rect = FancyBboxPatch((3.7, y-0.2), 0.6, 0.4, boxstyle="round,pad=0.02",
                              facecolor=COLORS['success_green'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(4.0, y, 'RY', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    # CX gates (entanglement to ancilla)
    cx_pairs = [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]
    for ctrl, targ in cx_pairs:
        ctrl_y = qubit_y[ctrl]
        targ_y = qubit_y[targ]
        # Control dot
        ax.plot(5.5 + ctrl*0.15, ctrl_y, 'o', color='black', markersize=6)
        # Target circle
        ax.plot(5.5 + ctrl*0.15, targ_y, 'o', color='black', markersize=10, fillstyle='none', linewidth=2)
        # Vertical line
        ax.plot([5.5 + ctrl*0.15, 5.5 + ctrl*0.15], [ctrl_y, targ_y], 'k-', linewidth=1.5)

    # Trainable Layer 2 (all qubits)
    for y in qubit_y:
        rect = FancyBboxPatch((7.7, y-0.2), 0.6, 0.4, boxstyle="round,pad=0.02",
                              facecolor=COLORS['success_green'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(8.0, y, 'RY', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    # Measurement on ancilla
    rect = FancyBboxPatch((9.6, qubit_y[6]-0.3), 0.8, 0.6, boxstyle="round,pad=0.02",
                          facecolor=COLORS['highlight_purple'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(10.0, qubit_y[6], 'M', ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    # Output annotation
    ax.annotate('P(|1⟩) = Dissimilarity', xy=(10.5, qubit_y[6]), xytext=(11, 2.5),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['highlight_purple'], lw=2),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['highlight_purple']))

    # Legend - positioned at far right bottom
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['quantum_blue'], edgecolor='black', label='v1 encoding (3 qubits)'),
        mpatches.Patch(facecolor=COLORS['warning_orange'], edgecolor='black', label='v2 encoding (3 qubits)'),
        mpatches.Patch(facecolor=COLORS['success_green'], edgecolor='black', label='Trainable layers (θ)'),
        mpatches.Patch(facecolor=COLORS['highlight_purple'], edgecolor='black', label='Ancilla (output)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=9)

    # Key properties box - moved to upper right
    props_text = """Key Properties:
• Sparse: 1 feature/qubit
• Scaling: [0.1, π-0.1]
• Parameters: 21 (7×3)
• Loss: BCE"""
    ax.text(11.5, 7, props_text, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig('figures/fig4_v3_architecture.png', facecolor='white')
    plt.savefig('figures/fig4_v3_architecture.pdf', facecolor='white')
    print("Saved: figures/fig4_v3_architecture.png/pdf")
    plt.close()


# =============================================================================
# FIGURE 5: Summary Dashboard
# =============================================================================
def plot_summary_dashboard():
    """Single-page summary of all key results."""
    fig = plt.figure(figsize=(14, 10))

    # Title
    fig.suptitle('Quantum Advantage in Semantic Geometry: Key Results Summary',
                 fontsize=16, fontweight='bold', y=0.98)

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Encoding Hierarchy
    ax1 = fig.add_subplot(gs[0, 0])
    encodings = ['DIRECT', 'CONCAT', 'DIFF']
    correlations = [0.989, 0.586, 0.007]
    colors = [COLORS['success_green'], COLORS['warning_orange'], COLORS['error_red']]
    ax1.bar(encodings, correlations, color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Correlation (r)')
    ax1.set_title('A) Encoding Hierarchy\n132× Gap', fontweight='bold', fontsize=11)
    ax1.set_ylim(0, 1.1)
    for i, (enc, corr) in enumerate(zip(encodings, correlations)):
        ax1.text(i, corr + 0.03, f'{corr:.3f}', ha='center', fontsize=9, fontweight='bold')

    # Panel 2: Entanglement Effect
    ax2 = fig.add_subplot(gs[0, 1])
    circuits = ['Entangled', 'Product']
    corrs = [0.68, -0.31]
    colors2 = [COLORS['quantum_blue'], COLORS['error_red']]
    ax2.bar(circuits, corrs, color=colors2, edgecolor='black', linewidth=1)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('Correlation (r)')
    ax2.set_title('B) Entanglement Effect\n+0.99 Advantage', fontweight='bold', fontsize=11)
    ax2.set_ylim(-0.5, 0.9)
    for i, corr in enumerate(corrs):
        offset = 0.03 if corr > 0 else -0.08
        ax2.text(i, corr + offset, f'{corr:+.2f}', ha='center', fontsize=9, fontweight='bold')

    # Panel 3: Hardware vs Simulator
    ax3 = fig.add_subplot(gs[0, 2])
    platforms = ['Hardware', 'Simulator']
    losses = [0.018, 0.211]
    colors3 = [COLORS['success_green'], COLORS['neutral_gray']]
    ax3.bar(platforms, losses, color=colors3, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Training Loss')
    ax3.set_title('C) Hardware Advantage\n93% Better', fontweight='bold', fontsize=11)
    for i, loss in enumerate(losses):
        ax3.text(i, loss + 0.01, f'{loss:.3f}', ha='center', fontsize=9, fontweight='bold')

    # Panel 4: Training Effect (Learning)
    ax4 = fig.add_subplot(gs[1, 0])
    states = ['Random', 'Trained']
    corrs4 = [-0.62, 0.68]
    colors4 = [COLORS['neutral_gray'], COLORS['success_green']]
    ax4.bar(states, corrs4, color=colors4, edgecolor='black', linewidth=1)
    ax4.axhline(y=0, color='black', linewidth=0.8)
    ax4.set_ylabel('Correlation (r)')
    ax4.set_title('D) Learning Effect\n+1.31 Training Effect', fontweight='bold', fontsize=11)
    ax4.set_ylim(-0.8, 0.9)
    for i, corr in enumerate(corrs4):
        offset = 0.03 if corr > 0 else -0.08
        ax4.text(i, corr + offset, f'{corr:+.2f}', ha='center', fontsize=9, fontweight='bold')

    # Panel 5: Ablation Summary (horizontal)
    ax5 = fig.add_subplot(gs[1, 1:])
    versions = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']
    effects = [-0.04, -0.02, 1.22, 0.46, -0.50, -0.08, 0.10]
    bar_colors = [COLORS['error_red'] if e < 0 else (COLORS['success_green'] if e > 0.5 else COLORS['warning_orange']) for e in effects]
    ax5.bar(versions, effects, color=bar_colors, edgecolor='black', linewidth=1)
    ax5.axhline(y=0, color='black', linewidth=1)
    ax5.set_ylabel('Training Effect')
    ax5.set_xlabel('Architecture Version')
    ax5.set_title('E) Seven-Version Ablation: V3 Wins (+1.22)', fontweight='bold', fontsize=11)
    ax5.set_ylim(-0.7, 1.4)
    for i, (v, e) in enumerate(zip(versions, effects)):
        offset = 0.05 if e > 0 else -0.1
        ax5.text(i, e + offset, f'{e:+.2f}', ha='center', fontsize=8, fontweight='bold')
    # Highlight V3
    ax5.annotate('WINNER', xy=(2, 1.22), xytext=(2, 1.35),
                fontsize=10, fontweight='bold', color=COLORS['success_green'], ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('figures/fig5_summary_dashboard.png', facecolor='white')
    plt.savefig('figures/fig5_summary_dashboard.pdf', facecolor='white')
    print("Saved: figures/fig5_summary_dashboard.png/pdf")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Generating publication figures for Paper 4...")
    print("=" * 50)

    plot_encoding_hierarchy()
    plot_quantum_advantage()
    plot_ablation_study()
    plot_v3_architecture()
    plot_summary_dashboard()

    print("=" * 50)
    print("All figures saved to figures/ directory")
    print("\nFigure list:")
    print("  1. fig1_encoding_hierarchy.png/pdf - The 132× encoding gap")
    print("  2. fig2_quantum_advantage.png/pdf  - Three quantum advantages")
    print("  3. fig3_ablation_study.png/pdf     - V1-V7 ablation results")
    print("  4. fig4_v3_architecture.png/pdf    - Winning circuit diagram")
    print("  5. fig5_summary_dashboard.png/pdf  - All results summary")
