import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("presentation_assets", exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#30363D'
plt.rcParams['axes.linewidth'] = 1.0

def create_gait_ablation_dark():
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor='#161B22')
    ax.set_facecolor('#161B22')
    
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'ROC-AUC']
    baseline_1d = [68.0, 64.29, 72.73, 69.23, 68.18]
    bilstm_bce = [72.0, 78.57, 63.64, 75.86, 75.32]
    evidential = [68.0, 57.14, 81.82, 66.67, 75.32]
    
    x = np.arange(len(metrics))
    width = 0.26
    
    r1 = ax.bar(x - width, baseline_1d, width, label='1D CNN Baseline', color='#4A90E2', edgecolor='#30363D')
    r2 = ax.bar(x, bilstm_bce, width, label='Proposed BiLSTM (BCE)', color='#50E3C2', edgecolor='#30363D')
    r3 = ax.bar(x + width, evidential, width, label='Proposed Evidential', color='#F5A623', edgecolor='#30363D')
    
    ax.set_ylabel('Score (%)', color='#8B949E', fontsize=10)
    ax.set_title('Gait Subject-Level Ablation Benchmark (N = 25)', color='#F0F6FC', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='#F0F6FC', fontsize=10, fontweight='bold')
    ax.set_ylim(45, 92)
    ax.tick_params(colors='#8B949E')
    ax.grid(axis='y', color='#21262D', linestyle='--', alpha=0.7)
    
    # Annotate Sensitivity Gain Callout
    ax.annotate('+14.3% GAIN', xy=(1, 78.57), xytext=(1, 86),
                ha='center', fontsize=8.5, fontweight='bold', color='#50E3C2',
                arrowprops=dict(arrowstyle='->', color='#50E3C2', lw=1.5))
    
    # Annotate Highest Specificity Callout
    ax.annotate('81.8% SPEC', xy=(2 + width, 81.82), xytext=(2 + width, 88),
                ha='center', fontsize=8.5, fontweight='bold', color='#F5A623',
                arrowprops=dict(arrowstyle='->', color='#F5A623', lw=1.5))
    
    for r in r1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 1, f'{h:.1f}', ha='center', va='bottom', color='#4A90E2', fontsize=7.5)
    for r in r2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 1, f'{h:.1f}', ha='center', va='bottom', color='#50E3C2', fontsize=7.5, fontweight='bold')
    for r in r3:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 1, f'{h:.1f}', ha='center', va='bottom', color='#F5A623', fontsize=7.5)

    ax.legend(facecolor='#0D1117', edgecolor='#30363D', labelcolor='#F0F6FC', loc='upper right', fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig('presentation_assets/slide10_ablation_dark.png', dpi=250, bbox_inches='tight', facecolor='#161B22')
    plt.close()

def create_voice_comparison_dark():
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor='#161B22')
    ax.set_facecolor('#161B22')
    
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'ROC-AUC']
    baseline_1d = [66.67, 100.0, 33.33, 75.0, 100.0]
    wav2vec2 = [66.67, 66.67, 66.67, 66.67, 77.78]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    r1 = ax.bar(x - width/2, baseline_1d, width, label='1D CNN Raw Waveform', color='#4A90E2', edgecolor='#30363D')
    r2 = ax.bar(x + width/2, wav2vec2, width, label='Proposed Wav2Vec2', color='#50E3C2', edgecolor='#30363D')
    
    ax.set_ylabel('Score (%)', color='#8B949E', fontsize=10)
    ax.set_title('Speech Modality Benchmark (N = 6 Held-Out Subjects)', color='#F0F6FC', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='#F0F6FC', fontsize=10, fontweight='bold')
    ax.set_ylim(20, 115)
    ax.tick_params(colors='#8B949E')
    ax.grid(axis='y', color='#21262D', linestyle='--', alpha=0.7)
    
    for r in r1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 2, f'{h:.1f}%', ha='center', va='bottom', color='#4A90E2', fontsize=8, fontweight='bold')
    for r in r2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 2, f'{h:.1f}%', ha='center', va='bottom', color='#50E3C2', fontsize=8, fontweight='bold')

    # Warning badge inside plot
    ax.text(2, 28, 'WARNING: 1D CNN suffers 33.3% Specificity (2/3 False Alarms)\nContinuous AUC = 1.000 reflects score rank separation',
            color='#E06C75', fontsize=8.5, ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0D1117', edgecolor='#E06C75', lw=1.2))

    ax.legend(facecolor='#0D1117', edgecolor='#30363D', labelcolor='#F0F6FC', loc='upper right', fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig('presentation_assets/slide11_voice_comparison_dark.png', dpi=250, bbox_inches='tight', facecolor='#161B22')
    plt.close()

def create_corruption_loco_dark():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2), facecolor='#161B22')
    ax1.set_facecolor('#161B22')
    ax2.set_facecolor('#161B22')
    
    # Left: LOCO Generalization
    cohorts = ['Ga (N=47)', 'Ju (N=54)', 'Si (N=64)', 'Macro Avg']
    acc_1d = [57.45, 72.22, 64.06, 64.58]
    acc_bi = [61.70, 77.78, 64.06, 67.85]
    
    x = np.arange(len(cohorts))
    w = 0.35
    r1 = ax1.bar(x - w/2, acc_1d, w, label='1D CNN', color='#4A90E2', edgecolor='#30363D')
    r2 = ax1.bar(x + w/2, acc_bi, w, label='Proposed BiLSTM', color='#50E3C2', edgecolor='#30363D')
    ax1.set_title('Cross-Hospital LOCO Accuracy', color='#F0F6FC', fontsize=11, fontweight='bold', pad=10)
    ax1.set_ylabel('Accuracy (%)', color='#8B949E', fontsize=9.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cohorts, color='#F0F6FC', fontsize=8.5)
    ax1.set_ylim(45, 88)
    ax1.tick_params(colors='#8B949E')
    ax1.grid(axis='y', color='#21262D', linestyle='--', alpha=0.7)
    ax1.legend(facecolor='#0D1117', edgecolor='#30363D', labelcolor='#F0F6FC', loc='upper left', fontsize=8)
    for r in r2:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2, h + 1, f'{h:.1f}%', ha='center', va='bottom', color='#50E3C2', fontsize=7.5, fontweight='bold')

    # Right: Expected Calibration Error under Sensor Corruption
    severities = ['Clean (0)', 'Low (1)', 'Med (2)', 'High (3)', 'Severe (4)']
    uncalibrated = [0.112, 0.145, 0.198, 0.254, 0.312]
    temp_scaled =  [0.054, 0.071, 0.092, 0.125, 0.165]
    deep_ens =     [0.038, 0.049, 0.068, 0.089, 0.118]
    
    ax2.plot(severities, uncalibrated, 'o--', color='#E06C75', lw=2.0, label='Uncalibrated (Base)')
    ax2.plot(severities, temp_scaled, 's-', color='#F5A623', lw=2.0, label='Temp Scaled (T*)')
    ax2.plot(severities, deep_ens, '^-', color='#50E3C2', lw=2.5, label='Deep Ensemble (M=5)')
    
    ax2.set_title('Calibration (ECE) vs Sensor Noise', color='#F0F6FC', fontsize=11, fontweight='bold', pad=10)
    ax2.set_ylabel('Expected Calibration Error (ECE)', color='#8B949E', fontsize=9.5)
    ax2.tick_params(colors='#8B949E')
    ax2.set_xticklabels(severities, color='#F0F6FC', fontsize=8)
    ax2.set_ylim(0, 0.35)
    ax2.grid(True, color='#21262D', linestyle='--', alpha=0.7)
    ax2.legend(facecolor='#0D1117', edgecolor='#30363D', labelcolor='#F0F6FC', loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('presentation_assets/slide12_corruption_loco_dark.png', dpi=250, bbox_inches='tight', facecolor='#161B22')
    plt.close()

def create_cross_modal_flow_dark():
    fig, ax = plt.subplots(figsize=(8.0, 4.4), facecolor='#161B22')
    ax.set_facecolor('#161B22')
    ax.axis('off')
    
    import matplotlib.patches as patches
    
    # 3 decision paths with generous spacing and high contrast
    cases = [
        {"y": 2.85, "status": "ACCEPT\n(High Trust)", "color": "#50E3C2",
         "cond": "• Modality Concordance: Gait P(PD)=0.89 & Voice P(PD)=0.79\n• Split Conformal Sets: Singleton {PD} on both modalities\n• Action: High-trust recommendation with 90% statistical coverage guarantee"},
        {"y": 1.55, "status": "FLAG\n(Medium Trust)", "color": "#F5A623",
         "cond": "• High Uncertainty / Boundary: Ensemble Var > 0.04 or Entropy > 0.65\n• Split Conformal Sets: Ambiguous Set {HC, PD}\n• Action: Secondary screening alert; deferred for standard clinical review"},
        {"y": 0.25, "status": "MODALITY\nCONFLICT\n(Triage)", "color": "#E06C75",
         "cond": "• Modality Discordance: Gait P(PD)=0.85 vs Voice P(PD)=0.35\n• Naive Average (0.60) REJECTED: Avoids false reassurance middle ground\n• Action: Immediate priority specialist referral with per-modality breakdown"}
    ]
    
    for c in cases:
        # Status Box
        sb = patches.FancyBboxPatch((0.2, c["y"]), 2.1, 1.1, boxstyle="round,pad=0.1",
                                    fc='#0D1117', ec=c["color"], lw=2.0)
        ax.add_patch(sb)
        ax.text(1.25, c["y"] + 0.55, c["status"], color=c["color"], fontsize=10, fontweight='bold', ha='center', va='center')
        
        # Details Box
        db = patches.FancyBboxPatch((2.55, c["y"]), 5.1, 1.1, boxstyle="round,pad=0.1",
                                    fc='#0D1117', ec='#30363D', lw=1.2)
        ax.add_patch(db)
        ax.text(2.75, c["y"] + 0.55, c["cond"], color='#F0F6FC', fontsize=8.5, ha='left', va='center')

    ax.text(3.9, 4.15, "DECISION-LEVEL RELIABILITY TRIAGE MATRIX", color='#50E3C2', fontsize=12, fontweight='bold', ha='center')
    ax.set_xlim(0, 7.8)
    ax.set_ylim(0, 4.4)
    
    plt.tight_layout()
    plt.savefig('presentation_assets/slide13_cross_modal_flow_dark.png', dpi=250, bbox_inches='tight', facecolor='#161B22')
    plt.close()

if __name__ == '__main__':
    create_gait_ablation_dark()
    create_voice_comparison_dark()
    create_corruption_loco_dark()
    create_cross_modal_flow_dark()
    print("Enhanced dark theme visual assets generated successfully!")
