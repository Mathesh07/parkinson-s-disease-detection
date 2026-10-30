import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("presentation_assets", exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#30363D'
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------------------------------------------
# 1. Slide 1 Hero Banner Graphic (Large, crisp, high contrast)
# -------------------------------------------------------------
def create_slide1_hero():
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#161B22')
    ax.set_facecolor('#161B22')
    ax.axis('off')
    
    # Draw simulated 16-channel gait waveforms
    t = np.linspace(0, 4 * np.pi, 400)
    for i in range(8):
        y = np.sin(t + i * 0.35) * np.exp(-0.04 * t) * (1 - i*0.07) + (i * 0.7)
        ax.plot(t, y - 3.2, color='#4A90E2', alpha=0.4 + 0.07*i, lw=2.0)
        
    # Draw simulated voice speech waveform
    t_v = np.linspace(0, 4 * np.pi, 400)
    y_v = (np.sin(3 * t_v) * np.cos(5 * t_v) + np.sin(11 * t_v) * 0.35) * 1.3 + 5.0
    ax.plot(t_v, y_v, color='#50E3C2', alpha=0.9, lw=2.2)
    ax.plot(t_v, y_v - 0.5, color='#50E3C2', alpha=0.4, lw=1.2)
    
    # Shield & Reliability Badge in center
    shield = patches.FancyBboxPatch((4.5, 0.4), 3.8, 3.2, boxstyle="round,pad=0.3", 
                                    fc='#0D1117', ec='#50E3C2', lw=2.5, zorder=5)
    ax.add_patch(shield)
    
    ax.text(6.4, 2.8, "RELIABILITY ENGINE", color='#50E3C2', fontsize=12, fontweight='bold', ha='center', zorder=6)
    ax.text(6.4, 2.1, "SAFETY VERIFIED", color='#F0F6FC', fontsize=15, fontweight='bold', ha='center', zorder=6)
    ax.text(6.4, 1.45, "• 90% Conformal Coverage\n• Dirichlet Evidential Logic\n• Cross-Modal Agreement Triage", 
            color='#C9D1D9', fontsize=9.5, ha='center', va='center', zorder=6)
    
    ax.text(0.5, -3.2, "16-Channel VGRF Gait Force Signals (100 Hz)", color='#58A6FF', fontsize=10.5, fontweight='bold')
    ax.text(0.5, 7.1, "Raw Acoustic Speech Waveforms (16 kHz)", color='#50E3C2', fontsize=10.5, fontweight='bold')
    
    ax.set_xlim(-0.5, 13.0)
    ax.set_ylim(-4.0, 8.0)
    
    plt.tight_layout()
    plt.savefig("presentation_assets/slide1_hero.png", dpi=250, bbox_inches='tight', facecolor='#161B22')
    plt.close()

# -------------------------------------------------------------
# 2. Slide 2: Reliability Gap Infographic
# -------------------------------------------------------------
def create_slide2_reliability_gap():
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor='#161B22')
    ax.set_facecolor('#161B22')
    ax.axis('off')
    
    # Left Box: Traditional AI (Red/Amber)
    p1 = patches.FancyBboxPatch((0.3, 0.4), 3.6, 3.6, boxstyle="round,pad=0.2",
                                fc='#0D1117', ec='#E06C75', lw=2.0)
    ax.add_patch(p1)
    ax.text(2.1, 3.5, "TRADITIONAL MEDICAL ML", color='#E06C75', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.1, 3.05, "(Standard Softmax Networks)", color='#8B949E', fontsize=9.5, ha='center')
    
    ax.text(0.6, 2.45, "• Forced Point Prediction:\n  Outputs P(PD) = 0.94 blindly", color='#F0F6FC', fontsize=9.5, fontweight='bold')
    ax.text(0.6, 1.80, "• Silent Failure Under Noise:\n  Overconfident on sensor faults", color='#C9D1D9', fontsize=9.0)
    ax.text(0.6, 1.20, "• Naive Probability Averaging:\n  Masks conflicting signals (0.60)", color='#C9D1D9', fontsize=9.0)
    ax.text(0.6, 0.65, "• RESULT: False reassurance & error", color='#E06C75', fontsize=9.5, fontweight='bold')

    # Right Box: Our Reliability Framework (Teal)
    p2 = patches.FancyBboxPatch((4.6, 0.4), 3.6, 3.6, boxstyle="round,pad=0.2",
                                fc='#0D1117', ec='#50E3C2', lw=2.0)
    ax.add_patch(p2)
    ax.text(6.4, 3.5, "OUR RELIABILITY SYSTEM", color='#50E3C2', fontsize=12, fontweight='bold', ha='center')
    ax.text(6.4, 3.05, "(Safety-First Evidential Layer)", color='#8B949E', fontsize=9.5, ha='center')
    
    ax.text(4.9, 2.45, "• Evidential Uncertainty:\n  Quantifies subjective u = K / S", color='#F0F6FC', fontsize=9.5, fontweight='bold')
    ax.text(4.9, 1.80, "• Split Conformal Sets:\n  Guarantees 90% true coverage", color='#C9D1D9', fontsize=9.0)
    ax.text(4.9, 1.20, "• Conflict Detection Policy:\n  Rejects naive averaging & flags", color='#C9D1D9', fontsize=9.0)
    ax.text(4.9, 0.65, "• RESULT: Safe specialist triage", color='#50E3C2', fontsize=9.5, fontweight='bold')

    # Arrow in middle
    ax.text(4.25, 2.2, "VS", color='#F5A623', fontsize=13, fontweight='bold', ha='center', va='center')
    
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 4.4)

    plt.tight_layout()
    plt.savefig("presentation_assets/slide2_reliability_gap.png", dpi=250, bbox_inches='tight', facecolor='#161B22')
    plt.close()

# -------------------------------------------------------------
# 3. Slide 4: System Architecture (Large, readable, high contrast)
# -------------------------------------------------------------
def create_slide4_architecture():
    fig, ax = plt.subplots(figsize=(11, 3.4), facecolor='#161B22')
    ax.set_facecolor('#161B22')
    ax.axis('off')
    
    stages = [
        {"x": 0.2, "w": 2.2, "title": "1. INPUT SENSORS", "color": "#58A6FF",
         "top_h": "Gait: 16-Ch VGRF\n(100 Hz, PhysioNet 165)", "bot_h": "Voice: 16 kHz Audio\n(MDVR-KCL 37 Subs)"},
        {"x": 2.8, "w": 2.5, "title": "2. NEURAL EXTRACTORS", "color": "#50E3C2",
         "top_h": "1D CNN + BiLSTM\n128-d Embedding", "bot_h": "Pre-trained Wav2Vec2\n768-d Mean Context"},
        {"x": 5.7, "w": 2.6, "title": "3. RELIABILITY ENGINES", "color": "#F5A623",
         "top_h": "Dirichlet Evidential (u)\nTemperature Scaling", "bot_h": "Split Conformal (90%)\nDeep Ensembles (M=5)"},
        {"x": 8.7, "w": 2.1, "title": "4. CLINICAL FUSION", "color": "#50E3C2",
         "top_h": "Decision Engine:\nACCEPT / FLAG / CONFLICT", "bot_h": "Streamlit Clinician UI\nInteractive Dashboard"}
    ]
    
    for s in stages:
        box = patches.FancyBboxPatch((s["x"], 0.15), s["w"], 3.0, boxstyle="round,pad=0.1",
                                     fc='#0D1117', ec=s["color"], lw=1.8)
        ax.add_patch(box)
        ax.text(s["x"] + s["w"]/2, 2.9, s["title"], color=s["color"], fontsize=11, fontweight='bold', ha='center')
        
        # Upper subcard
        c1 = patches.FancyBboxPatch((s["x"] + 0.12, 1.55), s["w"] - 0.24, 1.05, boxstyle="round,pad=0.08",
                                    fc='#21262D', ec='#30363D', lw=1.0)
        ax.add_patch(c1)
        ax.text(s["x"] + s["w"]/2, 2.05, s["top_h"], color='#F0F6FC', fontsize=9.5, fontweight='bold', ha='center', va='center')
        
        # Lower subcard
        c2 = patches.FancyBboxPatch((s["x"] + 0.12, 0.35), s["w"] - 0.24, 1.05, boxstyle="round,pad=0.08",
                                    fc='#21262D', ec='#30363D', lw=1.0)
        ax.add_patch(c2)
        ax.text(s["x"] + s["w"]/2, 0.85, s["bot_h"], color='#F0F6FC', fontsize=9.5, fontweight='bold', ha='center', va='center')

    # Bold arrows connecting stages
    ax.annotate("", xy=(2.75, 2.05), xytext=(2.45, 2.05), arrowprops=dict(arrowstyle="->", color='#58A6FF', lw=2.5))
    ax.annotate("", xy=(2.75, 0.85), xytext=(2.45, 0.85), arrowprops=dict(arrowstyle="->", color='#58A6FF', lw=2.5))
    ax.annotate("", xy=(5.65, 2.05), xytext=(5.35, 2.05), arrowprops=dict(arrowstyle="->", color='#50E3C2', lw=2.5))
    ax.annotate("", xy=(5.65, 0.85), xytext=(5.35, 0.85), arrowprops=dict(arrowstyle="->", color='#50E3C2', lw=2.5))
    ax.annotate("", xy=(8.65, 1.5), xytext=(8.35, 1.5), arrowprops=dict(arrowstyle="->", color='#F5A623', lw=2.5))

    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 3.4)

    plt.tight_layout()
    plt.savefig("presentation_assets/slide4_system_architecture.png", dpi=250, bbox_inches='tight', facecolor='#161B22')
    plt.close()

# -------------------------------------------------------------
# 4. Slide 6: Gait BiLSTM Tensor Architecture Flow (Readable)
# -------------------------------------------------------------
def create_slide6_gait_architecture():
    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor='#161B22')
    ax.set_facecolor('#161B22')
    ax.axis('off')
    
    layers = [
        {"name": "Input Window", "dim": "[B, 16, 500]", "sub": "16 Force Sensors\n5.0s @ 100Hz", "col": "#58A6FF"},
        {"name": "1D CNN Stages", "dim": "[B, 128, 125]", "sub": "2x Conv1D(k=5)\nBN + ReLU + MaxPool", "col": "#4A90E2"},
        {"name": "BiLSTM Sequence", "dim": "[B, 256]", "sub": "2-Layer Recurrence\nTemporal Mean Pool", "col": "#50E3C2"},
        {"name": "Dense Bottleneck", "dim": "[B, 128]", "sub": "Linear + BN + ReLU\nDropout (p=0.30)", "col": "#F5A623"},
        {"name": "Evidential Head", "dim": "[e_HC, e_PD]", "sub": "Dirichlet Evidence\nClosed u = K / S", "col": "#E06C75"}
    ]
    
    xs = [0.2, 2.4, 4.6, 6.8, 9.0]
    w = 1.9
    for i, (l, x) in enumerate(zip(layers, xs)):
        box = patches.FancyBboxPatch((x, 0.2), w, 2.3, boxstyle="round,pad=0.1",
                                     fc='#0D1117', ec=l["col"], lw=1.8)
        ax.add_patch(box)
        ax.text(x + w/2, 2.2, l["name"], color=l["col"], fontsize=11, fontweight='bold', ha='center')
        
        # Dim tag
        d_box = patches.FancyBboxPatch((x + 0.12, 1.45), w - 0.24, 0.48, boxstyle="round,pad=0.06",
                                       fc='#21262D', ec=l["col"], lw=1.0)
        ax.add_patch(d_box)
        ax.text(x + w/2, 1.68, l["dim"], color='#F0F6FC', fontsize=10, fontweight='bold', ha='center', va='center')
        
        ax.text(x + w/2, 0.8, l["sub"], color='#C9D1D9', fontsize=9, ha='center', va='center')
        
        if i < len(layers) - 1:
            ax.annotate("", xy=(xs[i+1] - 0.05, 1.4), xytext=(x + w + 0.05, 1.4),
                        arrowprops=dict(arrowstyle="->", color='#8B949E', lw=2.5))

    ax.set_xlim(0, 11.1)
    ax.set_ylim(0, 2.8)

    plt.tight_layout()
    plt.savefig("presentation_assets/slide6_gait_architecture.png", dpi=250, bbox_inches='tight', facecolor='#161B22')
    plt.close()

# -------------------------------------------------------------
# 5. Slide 7: Voice Wav2Vec2 Architecture Flow (Readable)
# -------------------------------------------------------------
def create_slide7_voice_architecture():
    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor='#161B22')
    ax.set_facecolor('#161B22')
    ax.axis('off')
    
    blocks = [
        {"name": "Raw Waveform", "dim": "16 kHz Mono", "desc": "Clinical Audio\nReadText / Dialogue", "col": "#58A6FF"},
        {"name": "CNN Latents", "dim": "7-Layer 1D CNN", "desc": "Temporal Encoder\nWaveform to Tokens", "col": "#4A90E2"},
        {"name": "Wav2Vec2 Base", "dim": "12 Transformers", "desc": "Contextual Attention\nPre-trained 100k+ hrs", "col": "#50E3C2"},
        {"name": "Mean Pooling", "dim": "768-dim Vector", "desc": "Temporal Summary\nover Chunk Tokens", "col": "#F5A623"},
        {"name": "Classification", "dim": "P(PD) per Chunk", "desc": "Linear(768->256->2)\nSubject Aggregation", "col": "#E06C75"}
    ]
    
    xs = [0.2, 2.4, 4.6, 6.8, 9.0]
    w = 1.9
    for i, (b, x) in enumerate(zip(blocks, xs)):
        box = patches.FancyBboxPatch((x, 0.2), w, 2.3, boxstyle="round,pad=0.1",
                                     fc='#0D1117', ec=b["col"], lw=1.8)
        ax.add_patch(box)
        ax.text(x + w/2, 2.2, b["name"], color=b["col"], fontsize=11, fontweight='bold', ha='center')
        
        d_box = patches.FancyBboxPatch((x + 0.12, 1.45), w - 0.24, 0.48, boxstyle="round,pad=0.06",
                                       fc='#21262D', ec=b["col"], lw=1.0)
        ax.add_patch(d_box)
        ax.text(x + w/2, 1.68, b["dim"], color='#F0F6FC', fontsize=10, fontweight='bold', ha='center', va='center')
        
        ax.text(x + w/2, 0.8, b["desc"], color='#C9D1D9', fontsize=9, ha='center', va='center')
        
        if i < len(blocks) - 1:
            ax.annotate("", xy=(xs[i+1] - 0.05, 1.4), xytext=(x + w + 0.05, 1.4),
                        arrowprops=dict(arrowstyle="->", color='#8B949E', lw=2.5))

    ax.set_xlim(0, 11.1)
    ax.set_ylim(0, 2.8)

    plt.tight_layout()
    plt.savefig("presentation_assets/slide7_voice_architecture.png", dpi=250, bbox_inches='tight', facecolor='#161B22')
    plt.close()

if __name__ == "__main__":
    create_slide1_hero()
    create_slide2_reliability_gap()
    create_slide4_architecture()
    create_slide6_gait_architecture()
    create_slide7_voice_architecture()
    print("Core slide visual diagrams updated with high readability and contrast!")
