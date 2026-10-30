import os
import sys
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib import colors

# Dimensions: 16:9 Widescreen Presentation (960 x 540 pt)
PAGE_W = 960
PAGE_H = 540

# Color Palette (Dark Theme matching repo aesthetics)
BG_DARK = colors.HexColor('#0D1117')
CARD_BG = colors.HexColor('#161B22')
CARD_BORDER = colors.HexColor('#30363D')
TEAL = colors.HexColor('#50E3C2')
BLUE = colors.HexColor('#58A6FF')
AMBER = colors.HexColor('#F5A623')
RED = colors.HexColor('#E06C75')
TEXT_MAIN = colors.HexColor('#F0F6FC')
TEXT_BODY = colors.HexColor('#C9D1D9')
TEXT_MUTED = colors.HexColor('#8B949E')

styles = getSampleStyleSheet()

# Custom typography styles
style_cat = ParagraphStyle(
    'CatStyle', parent=styles['Normal'],
    fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=TEAL
)
style_title = ParagraphStyle(
    'TitleStyle', parent=styles['Normal'],
    fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=TEXT_MAIN
)
style_sub = ParagraphStyle(
    'SubStyle', parent=styles['Normal'],
    fontName='Helvetica', fontSize=9.5, leading=13, textColor=TEXT_MUTED
)
style_card_title = ParagraphStyle(
    'CardTitleStyle', parent=styles['Normal'],
    fontName='Helvetica-Bold', fontSize=10.5, leading=14, textColor=TEAL
)
style_card_title_blue = ParagraphStyle(
    'CardTitleBlueStyle', parent=styles['Normal'],
    fontName='Helvetica-Bold', fontSize=10.5, leading=14, textColor=BLUE
)
style_card_title_amber = ParagraphStyle(
    'CardTitleAmberStyle', parent=styles['Normal'],
    fontName='Helvetica-Bold', fontSize=10.5, leading=14, textColor=AMBER
)
style_body = ParagraphStyle(
    'BodyStyle', parent=styles['Normal'],
    fontName='Helvetica', fontSize=8.5, leading=12.5, textColor=TEXT_BODY
)
style_body_bold = ParagraphStyle(
    'BodyBoldStyle', parent=styles['Normal'],
    fontName='Helvetica-Bold', fontSize=8.5, leading=12.5, textColor=TEXT_MAIN
)

def draw_background(c):
    c.setFillColor(BG_DARK)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

def draw_header(c, title, category, subtitle, slide_num, total_slides=15):
    # Category Pill
    p_cat = Paragraph(category.upper(), style_cat)
    p_cat.wrap(500, 15)
    p_cat.drawOn(c, 36, 508)
    
    # Title
    p_title = Paragraph(title, style_title)
    p_title.wrap(750, 25)
    p_title.drawOn(c, 36, 482)
    
    # Subtitle
    if subtitle:
        p_sub = Paragraph(subtitle, style_sub)
        p_sub.wrap(750, 16)
        p_sub.drawOn(c, 36, 465)
        
    # Slide Number Badge
    c.setFont('Helvetica-Bold', 10)
    c.setFillColor(TEXT_MUTED)
    num_str = f"{slide_num:02d} / {total_slides:02d}"
    c.drawRightString(PAGE_W - 36, 498, num_str)
    
    # Header divider line
    c.setStrokeColor(CARD_BORDER)
    c.setLineWidth(1.0)
    c.line(36, 458, PAGE_W - 36, 458)

def draw_card(c, x, y, w, h, border_color=CARD_BORDER, bg_color=CARD_BG, r=6):
    c.setFillColor(bg_color)
    c.setStrokeColor(border_color)
    c.setLineWidth(1.2)
    c.roundRect(x, y, w, h, r, fill=1, stroke=1)

def draw_card_with_text(c, x, y, w, h, title, content_html, title_color=TEAL, border_color=CARD_BORDER, bg_color=CARD_BG):
    """Draws a card and guarantees that text flows neatly directly under the card title."""
    draw_card(c, x, y, w, h, border_color=border_color, bg_color=bg_color)
    
    padding_x = 16
    padding_top = 16
    avail_w = w - (padding_x * 2)
    
    # Title
    t_style = ParagraphStyle('DynCardTitle', parent=style_card_title, textColor=title_color)
    p_t = Paragraph(f"<b>{title}</b>", t_style)
    w_t, h_t = p_t.wrap(avail_w, 25)
    
    title_top_y = y + h - padding_top
    title_draw_y = title_top_y - h_t
    p_t.drawOn(c, x + padding_x, title_draw_y)
    
    # Body
    b_top_y = title_draw_y - 10
    avail_h = b_top_y - (y + 12)
    
    p_b = Paragraph(content_html, style_body)
    w_b, h_b = p_b.wrap(avail_w, avail_h)
    p_b.drawOn(c, x + padding_x, b_top_y - h_b)

def draw_footer(c):
    c.setFont('Helvetica', 7.5)
    c.setFillColor(TEXT_MUTED)
    footer_text = "Reliability-Aware Parkinson's Disease Analysis from Gait & Speech  |  Academic Decision Support  |  PhysioNet & MDVR-KCL"
    c.drawString(36, 16, footer_text)

def render_deck():
    pdf_filename = "Parkinson_Disease_Detection_Presentation.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=(PAGE_W, PAGE_H))

    # =========================================================================
    # SLIDE 1: Title & Project Identity
    # =========================================================================
    draw_background(c)
    
    # Left Hero Title Card
    draw_card(c, 36, 36, 450, 468, border_color=TEAL)
    
    p = Paragraph("<font color='#50E3C2'><b>RESEARCH PROJECT PRESENTATION & VERIFICATION AUDIT</b></font>", style_cat)
    p.wrap(410, 20)
    p.drawOn(c, 56, 468)
    
    p_t = Paragraph("<b>Reliability-Aware Multimodal Analysis for Parkinson’s Disease Detection</b>", 
                    ParagraphStyle('HeroTitle', parent=style_title, fontSize=20, leading=24))
    p_t.wrap(410, 60)
    p_t.drawOn(c, 56, 405)
    
    p_sub = Paragraph("<b>Objective Biomechanical Gait & Acoustic Speech Analysis Under Epistemic Uncertainty</b>",
                      ParagraphStyle('HeroSub', parent=style_body, fontSize=10, leading=14, textColor=BLUE))
    p_sub.wrap(410, 30)
    p_sub.drawOn(c, 56, 370)
    
    content_s1 = (
        "<b>• Dual Clinical Modalities:</b> 16-channel Vertical Ground Reaction Force (VGRF) biomechanics "
        "from PhysioNet (165 subjects) and clinical acoustic speech recordings from MDVR-KCL (37 subjects).<br/><br/>"
        "<b>• Beyond Standard Classification:</b> Incorporating Dirichlet evidential neural heads, Deep Ensembles "
        "(M=5), Temperature Scaling, and Split Conformal Prediction (90% coverage guarantees).<br/><br/>"
        "<b>• Controlled 1D CNN Baselines:</b> Scientific benchmarking against fair raw-input 1D CNN models "
        "on identical subject partitions to isolate the value of temporal BiLSTM and Wav2Vec2 representations.<br/><br/>"
        "<b>• Research Prototype Framing:</b> Developed as an academic decision-support framework to flag "
        "clinical uncertainty; not intended as an autonomous medical diagnostic device."
    )
    p_b = Paragraph(content_s1, ParagraphStyle('HeroBody', parent=style_body, fontSize=9.0, leading=14.0))
    w_b, h_b = p_b.wrap(410, 300)
    p_b.drawOn(c, 56, 350 - h_b)

    # Right Hero Image Card
    draw_card(c, 502, 36, 422, 468, border_color=CARD_BORDER)
    c.drawImage("presentation_assets/slide1_hero.png", 517, 180, width=392, height=300, preserveAspectRatio=True)
    
    p_rt = Paragraph("<b>CORE SYSTEM ARCHITECTURE HIGHLIGHTS</b>", style_card_title)
    p_rt.wrap(390, 20)
    p_rt.drawOn(c, 522, 155)
    
    rt_content = (
        "<b>1. Gait Biomechanics:</b> 1D CNN + BiLSTM on 16 force channels (500 time steps @ 100 Hz).<br/><br/>"
        "<b>2. Speech Audio:</b> Pre-trained Wav2Vec2 fine-tuned on clinical speech tasks.<br/><br/>"
        "<b>3. Reliability Layer:</b> Closed-form Dirichlet uncertainty (u = K / S) and 90% conformal sets.<br/><br/>"
        "<b>4. Conflict Policy:</b> Decision-level fusion that detects and flags cross-modal disagreements."
    )
    p_rb = Paragraph(rt_content, ParagraphStyle('RBody', parent=style_body, fontSize=8.5, leading=13.0))
    w_rb, h_rb = p_rb.wrap(390, 130)
    p_rb.drawOn(c, 522, 140 - h_rb)
    
    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 2: Problem Statement & Motivation
    # =========================================================================
    draw_background(c)
    draw_header(c, "The Reliability Gap in Medical Machine Learning", "PROBLEM STATEMENT & MOTIVATION",
                "Why conventional softmax classifiers fail in clinical neurodegenerative screening", 2)

    s2_left = (
        "<b>1. Clinical Nuance & Symptom Overlap:</b><br/>"
        "Parkinsonian motor degradation causes bilateral gait asymmetry and subtle vocal hypophonia/dysarthria "
        "that overlap significantly with normal healthy aging, requiring high sensitivity without excessive false alarms.<br/><br/>"
        "<b>2. The Silent Overconfidence Failure Mode:</b><br/>"
        "Standard neural networks trained with cross-entropy force uncalibrated point probabilities, outputting "
        "&gt;95% confidence on out-of-distribution inputs and corrupted sensor streams without flagging doubt.<br/><br/>"
        "<b>3. Inter-Cohort Distribution Shift:</b><br/>"
        "Sensor shoe designs, floor plates, acoustic hardware, and speaking protocols vary across clinical sites, "
        "causing models that look accurate on one hospital cohort to degrade sharply on unseen clinics.<br/><br/>"
        "<b>4. The Hazard of Naive Multimodal Fusion:</b><br/>"
        "Averaging conflicting predictions (e.g., Gait P(PD)=0.85 vs Voice P(PD)=0.35) yields a false middle ground "
        "(0.60) that silently obscures severe sensor or task conflict."
    )
    draw_card_with_text(c, 36, 36, 430, 408, "CLINICAL & STATISTICAL FAILURE MODES", s2_left, title_color=BLUE)

    # Right Column: Visual Diagram & Framework Solution
    draw_card(c, 482, 36, 442, 408)
    c.drawImage("presentation_assets/slide2_reliability_gap.png", 497, 185, width=412, height=240, preserveAspectRatio=True)

    p_rt2 = Paragraph("<b>OUR SAFETY-FIRST TRIAGE SOLUTION</b>", style_card_title)
    p_rt2.wrap(400, 20)
    p_rt2.drawOn(c, 502, 155)

    s2_right = (
        "<b>• Evidential Epistemic Heads:</b> Parameterize Dirichlet distributions Dir(&alpha;_k) to quantify subjective "
        "uncertainty in a single forward pass without expensive sampling.<br/><br/>"
        "<b>• Finite-Sample Conformal Sets:</b> Outputs statistical sets ({PD}, {HC}, or {HC, PD}) guaranteeing "
        "90% true-label coverage, converting forced guesses into actionable clinical ambiguity flags.<br/><br/>"
        "<b>• Conflict Detection Rules:</b> Explicitly triggers a MODALITY_CONFLICT alert when gait and voice "
        "diverge, safely directing the case to specialist human review."
    )
    p_rb2 = Paragraph(s2_right, ParagraphStyle('S2Right', parent=style_body, fontSize=8.5, leading=13.0))
    w_rb2, h_rb2 = p_rb2.wrap(400, 130)
    p_rb2.drawOn(c, 502, 140 - h_rb2)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 3: Objectives & Proposed Technical Contributions (Grid Layout)
    # =========================================================================
    draw_background(c)
    draw_header(c, "Project Objectives & Technical Contributions", "CORE CONTRIBUTIONS",
                "Five concrete engineering contributions implemented and verified in the repository", 3)

    contributions_grid = [
        # Top row: 3 cards
        {"x": 36, "y": 245, "w": 282, "h": 198, "col": TEAL,
         "title": "1. Temporal Biomechanics (Gait)",
         "desc": "<b>End-to-End 1D CNN + BiLSTM</b><br/><br/>"
                 "Developed an end-to-end architecture on 16 continuous force channels to capture both intra-stride "
                 "pressure impulses and long-range inter-stride bilateral asymmetry across 500-step windows."},
        {"x": 339, "y": 245, "w": 282, "h": 198, "col": BLUE,
         "title": "2. Speech Representation (Voice)",
         "desc": "<b>Fine-Tuned Wav2Vec2 Audio</b><br/><br/>"
                 "Fine-tuned pre-trained self-supervised Wav2Vec2 representations on clinical speech tasks, "
                 "benchmarked against a newly developed raw-waveform 1D CNN baseline on identical splits."},
        {"x": 642, "y": 245, "w": 282, "h": 198, "col": AMBER,
         "title": "3. Evidential Uncertainty Logic",
         "desc": "<b>Subjective Logic Dirichlet Heads</b><br/><br/>"
                 "Implemented evidential neural heads parameterizing Dirichlet distributions. Delivers non-negative "
                 "evidence and closed-form epistemic uncertainty (u = K / S) in a single deterministic pass."},
        # Bottom row: 2 cards
        {"x": 36, "y": 36, "w": 435, "h": 195, "col": BLUE,
         "title": "4. Cross-Site Generalization & Sensor Corruption Benchmarking",
         "desc": "<b>Leave-One-Cohort-Out (LOCO) Robustness</b><br/><br/>"
                 "Evaluated models across three independent hospital sites (165 subjects) and stress-tested "
                 "calibration stability (Expected Calibration Error) under Gaussian sensor noise, channel dropout, "
                 "and force signal attenuation."},
        {"x": 489, "y": 36, "w": 435, "h": 195, "col": TEAL,
         "title": "5. Safety-First Cross-Modal Fusion & Clinician Dashboard",
         "desc": "<b>Decision-Level Agreement Policy & Live UI</b><br/><br/>"
                 "Engineered a decision-level reliability fusion engine that outputs ACCEPT, FLAG, or CONFLICT "
                 "statuses, integrated into an interactive Streamlit demonstration with live transparency panels."}
    ]

    for item in contributions_grid:
        draw_card_with_text(c, item["x"], item["y"], item["w"], item["h"],
                           item["title"], item["desc"], title_color=item["col"], border_color=item["col"])

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 4: Complete System Architecture
    # =========================================================================
    draw_background(c)
    draw_header(c, "End-to-End Reliability System Architecture", "SYSTEM PIPELINE",
                "Complete dataflow from raw biomechanics & acoustic signals to reliability triage", 4)

    # Top Card: Architecture Diagram
    draw_card(c, 36, 215, 888, 228)
    c.drawImage("presentation_assets/slide4_system_architecture.png", 46, 225, width=868, height=208, preserveAspectRatio=True)

    # Bottom 3 Explanatory Cards
    c_w = 282
    c_h = 165
    
    b1_text = (
        "<b>• Gait Stream:</b> 16-channel VGRF (100 Hz) &rarr; 5.0 s windows &rarr; 1D CNN + BiLSTM &rarr; 128-d dense embedding.<br/><br/>"
        "<b>• Voice Stream:</b> 16 kHz Mono &rarr; 3.0 s chunks &rarr; Wav2Vec2 Contextual Transformer &rarr; 768-d mean vector."
    )
    draw_card_with_text(c, 36, 36, c_w, c_h, "1. DUAL FEATURE EXTRACTORS", b1_text, title_color=BLUE, border_color=BLUE)

    b2_text = (
        "<b>• Dirichlet Evidential:</b> Single-pass subjective uncertainty u = K / S.<br/><br/>"
        "<b>• Conformal Prediction:</b> Split conformal sets guaranteeing 90% true coverage.<br/><br/>"
        "<b>• Deep Ensembles:</b> M=5 independent seeds capturing epistemic variance."
    )
    draw_card_with_text(c, 339, 36, c_w, c_h, "2. RELIABILITY VERIFICATION", b2_text, title_color=AMBER, border_color=AMBER)

    b3_text = (
        "<b>• High Concordance:</b> Singleton overlap &rarr; <b>ACCEPT</b>.<br/><br/>"
        "<b>• Ambiguous Bounds:</b> High entropy/variance &rarr; <b>FLAG</b>.<br/><br/>"
        "<b>• Cross-Modal Conflict:</b> Opposing predictions &rarr; <b>CONFLICT</b>.<br/><br/>"
        "<b>• Delivery:</b> Interactive Streamlit clinician dashboard."
    )
    draw_card_with_text(c, 642, 36, c_w, c_h, "3. DECISION-LEVEL FUSION", b3_text, title_color=TEAL, border_color=TEAL)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 5: Datasets & Zero-Leakage Preprocessing
    # =========================================================================
    draw_background(c)
    draw_header(c, "Clinical Datasets & Zero-Leakage Preprocessing Protocol", "DATASET & SPLITTING PROTOCOL",
                "PhysioNet (165 subjects) and MDVR-KCL (37 subjects) under strict patient-level separation", 5)

    d_text = (
        "<b>Gait Dataset (PhysioNet - Gait in Parkinson's Disease):</b><br/>"
        "• <b>Total Cohort:</b> 165 subjects (93 PD, 72 Healthy Controls)<br/>"
        "• <b>Clinical Sites:</b> Galit (Ga: 47), Hausdorff (Ju: 54), Silberstein (Si: 64)<br/>"
        "• <b>Sensor Setup:</b> 16 continuous Vertical Ground Reaction Force (VGRF) sensors "
        "(8 Left foot, 8 Right foot) sampled at 100 Hz.<br/>"
        "• <b>Temporal Windowing:</b> 5.0 s duration (500 time steps) with 50% overlap (2.5 s step).<br/>"
        "• <b>Evaluated Windows:</b> 14,440 continuous biomechanical samples.<br/><br/>"
        "<b>Voice Dataset (MDVR-KCL - King's College London):</b><br/>"
        "• <b>Total Cohort:</b> 37 subjects (22 PD, 15 Healthy Controls)<br/>"
        "• <b>Acoustic Tasks:</b> ReadText and SpontaneousDialogue recordings.<br/>"
        "• <b>Audio Format:</b> 16 kHz single-channel float32 waveforms.<br/>"
        "• <b>Segmentation:</b> Overlapping temporal chunks aggregated to patient level."
    )
    draw_card_with_text(c, 36, 36, 430, 408, "CLINICAL DATASET SPECIFICATIONS", d_text, title_color=TEAL)

    z_text = (
        "<b>Strict Subject-Level Stratification (Seed 42):</b><br/>"
        "• <b>Zero Patient Overlap:</b> No individual appears in more than one partition:<br/>"
        "  &nbsp;&nbsp;&nbsp;&nbsp;Train &cap; Val = &empty;,&nbsp;&nbsp;Train &cap; Test = &empty;,&nbsp;&nbsp;Val &cap; Test = &empty;.<br/>"
        "• <b>Gait Partitions:</b> 115 Train, 25 Validation, 25 Held-Out Test subjects (11 HC, 14 PD).<br/>"
        "• <b>Voice Partitions:</b> 26 Train, 5 Validation, 6 Held-Out Test subjects (3 HC, 3 PD).<br/><br/>"
        "<b>Strict Train-Only Normalizer Estimation:</b><br/>"
        "• Channel-wise mean and standard deviation for Z-score normalization are computed "
        "strictly on training subjects.<br/>"
        "• Validation and test subjects are normalized using frozen training parameters.<br/><br/>"
        "<b>Elimination of Contamination:</b><br/>"
        "Completely prevents window-level and recording-level leakage where overlapping windows "
        "from the same patient pollute both train and test sets—a rampant flaw in published literature."
    )
    draw_card_with_text(c, 482, 36, 442, 408, "ZERO-LEAKAGE SCIENTIFIC GUARANTEE", z_text, title_color=BLUE)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 6: Gait Model: 1D CNN + BiLSTM
    # =========================================================================
    draw_background(c)
    draw_header(c, "Gait Biomechanical Modeling (1D CNN + BiLSTM)", "GAIT ARCHITECTURE & TENSOR FLOW",
                "Capturing local pressure impulses and bilateral inter-stride temporal phase transitions", 6)

    # Top Card: Tensor Architecture Flow
    draw_card(c, 36, 215, 888, 228)
    c.drawImage("presentation_assets/slide6_gait_architecture.png", 46, 225, width=868, height=208, preserveAspectRatio=True)

    # Bottom 3 Cards: Architectural Stages
    s1_text = (
        "<b>• Input Tensor:</b> [B, 16, 500] (16 force sensors, 500 time steps).<br/><br/>"
        "<b>• Block 1:</b> Conv1D(16&rarr;64, k=5, p=2) &rarr; BN &rarr; ReLU &rarr; MaxPool(2) &rarr; [B, 64, 250].<br/><br/>"
        "<b>• Block 2:</b> Conv1D(64&rarr;128, k=5, p=2) &rarr; BN &rarr; ReLU &rarr; MaxPool(2) &rarr; [B, 128, 125].<br/><br/>"
        "<b>• Role:</b> Extracts local pressure impulses across foot sensors."
    )
    draw_card_with_text(c, 36, 36, c_w, c_h, "STAGE 1: 1D CONVOLUTIONS", s1_text, title_color=BLUE, border_color=BLUE)

    s2_text = (
        "<b>• Transpose:</b> [B, 125, 128] (Sequence Length, Features).<br/><br/>"
        "<b>• 2-Layer BiLSTM:</b> Hidden size 128 per direction &rarr; [B, 125, 256].<br/><br/>"
        "<b>• Global Pooling:</b> Temporal mean pooling &rarr; [B, 256].<br/><br/>"
        "<b>• Critical Role:</b> Tracks long-range bilateral stride phase transitions and timing degradation."
    )
    draw_card_with_text(c, 339, 36, c_w, c_h, "STAGE 2: BIDIRECTIONAL LSTM", s2_text, title_color=TEAL, border_color=TEAL)

    s3_text = (
        "<b>• Dense Bottleneck:</b> Linear(256&rarr;128) &rarr; BN &rarr; ReLU &rarr; Dropout(0.30).<br/><br/>"
        "<b>• BCE Head:</b> Linear(128&rarr;1) + Sigmoid.<br/><br/>"
        "<b>• Evidential Head:</b> Linear(128&rarr;2) + Softplus &rarr; Non-negative evidence [e_HC, e_PD] &ge; 0.<br/><br/>"
        "<b>• Closed Uncertainty:</b> u = 2 / (e_HC + e_PD + 2)."
    )
    draw_card_with_text(c, 642, 36, c_w, c_h, "STAGE 3: DENSE & EVIDENTIAL", s3_text, title_color=AMBER, border_color=AMBER)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 7: Speech Model: Wav2Vec2 Pipeline
    # =========================================================================
    draw_background(c)
    draw_header(c, "Speech Representation Learning (Wav2Vec2)", "VOICE ARCHITECTURE & TRANSFER LEARNING",
                "Self-supervised acoustic representation learning fine-tuned on clinical vocal recordings", 7)

    # Top Card: Wav2Vec2 Flow
    draw_card(c, 36, 215, 888, 228)
    c.drawImage("presentation_assets/slide7_voice_architecture.png", 46, 225, width=868, height=208, preserveAspectRatio=True)

    # Bottom 3 Cards
    v1_text = (
        "<b>• Pre-trained Corpus:</b> facebook/wav2vec2-base pre-trained on 100k+ hours of unlabelled speech.<br/><br/>"
        "<b>• 7-Layer CNN Encoder:</b> Temporal convolution extracts latent acoustic tokens from raw 16 kHz waveforms.<br/><br/>"
        "<b>• 12 Transformer Blocks:</b> Contextualizes acoustic representations over continuous speech."
    )
    draw_card_with_text(c, 36, 36, c_w, c_h, "1. PRE-TRAINED BACKBONE", v1_text, title_color=BLUE, border_color=BLUE)

    v2_text = (
        "<b>• Temporal Mean Pooling:</b> Averages contextual tokens across 3.0 s chunks into a 768-dim vector.<br/><br/>"
        "<b>• Classifier Head:</b> Linear(768&rarr;256) &rarr; ReLU &rarr; Dropout(0.30) &rarr; Linear(256&rarr;2).<br/><br/>"
        "<b>• Training Dynamics:</b> Fine-tuned with gradient accumulation (Batch=2, Accum=2, LR=10^-5) under Cross-Entropy."
    )
    draw_card_with_text(c, 339, 36, c_w, c_h, "2. DOWNSTREAM FINE-TUNING", v2_text, title_color=TEAL, border_color=TEAL)

    v3_text = (
        "<b>• Chunk Level:</b> Generates posterior probabilities P(PD) on overlapping 3.0 s segments.<br/><br/>"
        "<b>• Recording Level:</b> Averages chunk posteriors across individual ReadText or SpontaneousDialogue tasks.<br/><br/>"
        "<b>• Patient Level:</b> Mean of task posteriors forms the final clinical probability score."
    )
    draw_card_with_text(c, 642, 36, c_w, c_h, "3. SUBJECT-LEVEL INFERENCE", v3_text, title_color=AMBER, border_color=AMBER)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 8: Reliability Mechanisms: 5-Pillar Framework (Grid Layout)
    # =========================================================================
    draw_background(c)
    draw_header(c, "Safety & Reliability Mechanisms: 5-Pillar Framework", "RELIABILITY MECHANISMS",
                "Multi-layered uncertainty quantification, probability calibration, and finite-sample guarantees", 8)

    pillars_grid = [
        # Top row: 3 cards
        {"x": 36, "y": 245, "w": 282, "h": 198, "col": TEAL,
         "title": "1. Dirichlet Evidential Networks",
         "desc": "<b>Subjective Logic & Closed-Form Uncertainty</b><br/><br/>"
                 "Treats network outputs as evidence e_k &ge; 0 parameterizing a Dirichlet distribution Dir(&alpha;_k). "
                 "Closed-form epistemic uncertainty: u = K / S is computed in a single pass without sampling."},
        {"x": 339, "y": 245, "w": 282, "h": 198, "col": BLUE,
         "title": "2. Deep Ensembles (M = 5)",
         "desc": "<b>Multi-Seed Predictive Variance</b><br/><br/>"
                 "Trains 5 independent neural networks initialized with distinct random seeds. Disagreement is "
                 "quantified via predictive variance &sigma;&sup2;_ens and mutual information between member predictions."},
        {"x": 642, "y": 245, "w": 282, "h": 198, "col": BLUE,
         "title": "3. Monte Carlo Dropout (T = 20)",
         "desc": "<b>Stochastic Bayesian Sampling</b><br/><br/>"
                 "Preserves dropout masks (p = 0.30) during test-time inference across T=20 stochastic forward "
                 "passes to approximate the Bayesian posterior distribution and model parameter uncertainty."},
        # Bottom row: 2 cards
        {"x": 36, "y": 36, "w": 435, "h": 195, "col": AMBER,
         "title": "4. Temperature Scaling Probability Calibration",
         "desc": "<b>Post-Hoc Negative Log-Likelihood Optimization</b><br/><br/>"
                 "Optimizes a single scalar parameter T* on validation logits (z / T*) to minimize Expected Calibration "
                 "Error (ECE), eliminating systematic overconfidence without altering ROC-AUC rank ordering."},
        {"x": 489, "y": 36, "w": 435, "h": 195, "col": TEAL,
         "title": "5. Split Conformal Prediction (1 - α = 0.90 Coverage)",
         "desc": "<b>Distribution-Free Finite-Sample Guarantees</b><br/><br/>"
                 "Uses non-conformity scores s_i = 1 - P(y_i|x_i) on calibration subjects to compute exact empirical "
                 "quantiles. Guarantees 90% statistical coverage: singleton ({PD}), ambiguous ({HC, PD}), or empty sets."}
    ]

    for item in pillars_grid:
        draw_card_with_text(c, item["x"], item["y"], item["w"], item["h"],
                           item["title"], item["desc"], title_color=item["col"], border_color=item["col"])

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 9: Experimental Design & Controlled Baseline Framework
    # =========================================================================
    draw_background(c)
    draw_header(c, "Experimental Design & Controlled Baseline Framework", "EXPERIMENTAL DESIGN",
                "Establishing strict raw-input 1D CNN baselines and archiving literature 2D CNN references", 9)

    m_text = (
        "<b>Scientific Integrity Principle:</b><br/>"
        "To scientifically isolate the benefit of BiLSTM recurrent sequence modeling and Wav2Vec2 "
        "self-supervised representations, baseline models must operate on the exact same raw inputs "
        "and identical subject-level data partitions.<br/><br/>"
        "<b>Active Primary Baselines Created:</b><br/>"
        "• <b>Gait 1D CNN Baseline:</b> Pure 3-stage Conv1D (16&rarr;64&rarr;128&rarr;256) + Global Average Pooling "
        "+ Linear Classifier. Operates directly on raw 16-channel, 500-step VGRF windows (no LSTM, no evidential head).<br/>"
        "• <b>Voice 1D CNN Baseline:</b> Pure 4-stage Conv1D (1&rarr;32&rarr;64&rarr;128&rarr;256) + Adaptive Pooling "
        "on raw 16 kHz waveforms (no pre-trained weights, no STFT).<br/><br/>"
        "<b>Archived Literature 2D CNN Models:</b><br/>"
        "• The paper-inspired models (Vásquez-Correa et al., 2019) used 2D CNNs on STFT representations.<br/>"
        "• We preserved these in Gait/baselines/base_paper_cnn/ and Voice/baselines/base_paper_cnn/ as archived "
        "supplementary references; excluded from active comparisons."
    )
    draw_card_with_text(c, 36, 36, 430, 408, "THE FAIR COMPARISON MANDATE", m_text, title_color=TEAL)

    mx_text = (
        "<b>GAIT MODALITY BENCHMARK (165 Subjects):</b><br/>"
        "• <b>Partitions:</b> 115 Train | 25 Validation | 25 Held-Out Test (11 HC, 14 PD)<br/>"
        "• <b>Input Data:</b> Raw 16-channel continuous force signals (500 steps @ 100 Hz)<br/>"
        "• <b>Baseline:</b> 1D CNN (Active Primary Benchmark)<br/>"
        "• <b>Proposed Models:</b> 1D CNN + BiLSTM (BCE) & Dirichlet Evidential Model<br/>"
        "• <b>Evaluated Metrics:</b> Accuracy, Balanced Acc, Sensitivity, Specificity, F1, ROC-AUC.<br/><br/>"
        "<b>VOICE MODALITY BENCHMARK (37 Subjects):</b><br/>"
        "• <b>Partitions:</b> 26 Train | 5 Validation | 6 Held-Out Test (3 HC, 3 PD)<br/>"
        "• <b>Input Data:</b> Raw 16 kHz audio waveforms<br/>"
        "• <b>Baseline:</b> 1D CNN Raw Waveform (Active Primary Benchmark)<br/>"
        "• <b>Proposed Model:</b> Pre-trained Wav2Vec2 + Contextual Classification Head<br/><br/>"
        "<b>AUDIT PROTOCOL GUARANTEE:</b><br/>"
        "Identical random seed (42), frozen train normalizers, subject-level prediction aggregation, "
        "and 95% bootstrap confidence intervals computed using subject-level resampling."
    )
    draw_card_with_text(c, 482, 36, 442, 408, "CONTROLLED EVALUATION MATRIX", mx_text, title_color=BLUE)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 10: Main Gait Results & Controlled Ablation (Novelty vs 1D CNN)
    # =========================================================================
    draw_background(c)
    draw_header(c, "Gait Evaluation: Baseline vs. Proposed Models", "GAIT EVALUATION & CONTROLLED ABLATION",
                "Independently recalculated subject-level metrics on 25 held-out test subjects (11 HC, 14 PD)", 10)

    ga_text = (
        "<b>1D CNN Baseline (Active Benchmark):</b><br/>"
        "• Accuracy: <b>68.0%</b> &nbsp;|&nbsp; Balanced Accuracy: <b>68.51%</b><br/>"
        "• Sensitivity: <b>64.29%</b> &nbsp;|&nbsp; Specificity: <b>72.73%</b> &nbsp;|&nbsp; F1: <b>69.23%</b><br/>"
        "• ROC-AUC: <b>0.6818</b> &nbsp;|&nbsp; Brier: 0.2592 &nbsp;|&nbsp; NLL: 1.2645<br/>"
        "• 95% Bootstrap CI: Acc [0.480, 0.880], AUC [0.462, 0.909]<br/><br/>"
        "<b>Proposed 1D CNN + BiLSTM (BCE Loss):</b><br/>"
        "• Accuracy: <b>72.0% (+4.0%)</b> &nbsp;|&nbsp; Balanced Acc: <b>71.10%</b><br/>"
        "• Sensitivity: <font color='#50E3C2'><b>78.57% (+14.28% gain!)</b></font> &nbsp;|&nbsp; Specificity: <b>63.64%</b><br/>"
        "• ROC-AUC: <font color='#50E3C2'><b>0.7532 (+0.0714 gain!)</b></font> &nbsp;|&nbsp; F1: <b>75.86%</b><br/>"
        "• 95% Bootstrap CI: Acc [0.520, 0.880], AUC [0.535, 0.942]<br/><br/>"
        "<b>Proposed Evidential Model (Subjective Logic):</b><br/>"
        "• Accuracy: <b>68.0%</b> &nbsp;|&nbsp; Specificity: <font color='#F5A623'><b>81.82% (Highest Spec!)</b></font><br/>"
        "• Sensitivity: <b>57.14%</b> &nbsp;|&nbsp; ROC-AUC: <b>0.7532</b> &nbsp;|&nbsp; F1: <b>66.67%</b><br/><br/>"
        "<b>NOVELTY & ABLATION TAKEAWAYS:</b><br/>"
        "<b>1. BiLSTM Recurrent Value:</b> Boosts detection sensitivity by +14.3% by capturing long-range stride timing.<br/>"
        "<b>2. Evidential Safety:</b> Cuts false positive alerts in half (2 false alarms vs 4), protecting clinical workflow."
    )
    draw_card_with_text(c, 36, 36, 420, 408, "AUDITED GAIT BENCHMARK (N = 25)", ga_text, title_color=TEAL)

    # Right Column: Visual Dark Plot
    draw_card(c, 472, 36, 452, 408)
    p_rp = Paragraph("<b>ABLATION METRICS COMPARISON (1D CNN vs BiLSTM vs Evidential)</b>", style_card_title_blue)
    p_rp.wrap(420, 18)
    p_rp.drawOn(c, 488, 418)
    c.drawImage("presentation_assets/slide10_ablation_dark.png", 482, 50, width=432, height=350, preserveAspectRatio=True)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 11: Speech Evaluation & Sample Size Reality
    # =========================================================================
    draw_background(c)
    draw_header(c, "Voice Evaluation: 1D CNN vs. Wav2Vec2", "VOICE EVALUATION & SAMPLE SIZE AUDIT",
                "Subject-level benchmark on 6 held-out test subjects and sample size audit", 11)

    va_text = (
        "<b>Raw-Waveform 1D CNN Baseline:</b><br/>"
        "• Accuracy: <b>66.67% (4/6)</b> &nbsp;|&nbsp; Sensitivity: <b>100.0% (3/3)</b><br/>"
        "• Specificity: <font color='#E06C75'><b>33.33% (1/3)</b></font> &nbsp;|&nbsp; F1: <b>75.0%</b> &nbsp;|&nbsp; ROC-AUC: <b>1.000</b><br/>"
        "• Catches all 3 PD patients but false-alarms on 2 out of 3 healthy controls.<br/><br/>"
        "<b>Proposed Pre-trained Wav2Vec2 Model:</b><br/>"
        "• Accuracy: <b>66.67% (4/6)</b> &nbsp;|&nbsp; Sensitivity: <b>66.67% (2/3)</b><br/>"
        "• Specificity: <b>66.67% (2/3)</b> &nbsp;|&nbsp; F1: <b>66.67%</b> &nbsp;|&nbsp; ROC-AUC: <b>0.7778</b><br/>"
        "• Provides a clinically balanced operating profile across PD and HC cohorts.<br/><br/>"
        "<b>EXPLAINING THE 1.000 ROC-AUC SCORE:</b><br/>"
        "• In the 1D CNN, all 3 PD subjects received continuous scores strictly higher (0.745, 0.747, 0.825) "
        "than all 3 HC subjects (0.131, 0.529, 0.639).<br/>"
        "• Because continuous rank ordering separates the classes perfectly, ROC-AUC is mathematically 1.000, "
        "even though thresholding at &tau; = 0.50 produces two false alarms.<br/><br/>"
        "<b>SCIENTIFIC INTEGRITY DISCLAIMER:</b><br/>"
        "With N = 6 subjects, each prediction alters accuracy by &plusmn;16.7%. These results demonstrate feasibility, "
        "not statistical finality."
    )
    draw_card_with_text(c, 36, 36, 420, 408, "AUDITED SPEECH BENCHMARK (N = 6)", va_text, title_color=TEAL)

    # Right Column: Visual Plot
    draw_card(c, 472, 36, 452, 408)
    p_vp = Paragraph("<b>VOICE MODELS BENCHMARK & SPECIFICITY ALERT</b>", style_card_title_amber)
    p_vp.wrap(420, 18)
    p_vp.drawOn(c, 488, 418)
    c.drawImage("presentation_assets/slide11_voice_comparison_dark.png", 482, 50, width=432, height=350, preserveAspectRatio=True)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 12: Generalization & Stress-Testing Under Distribution Shift
    # =========================================================================
    draw_background(c)
    draw_header(c, "Generalization & Stress-Testing Under Distribution Shift", "ROBUSTNESS & STRESS-TESTING",
                "Leave-One-Cohort-Out cross-site validation (165 subjects) and sensor corruption experiments", 12)

    lo_text = (
        "<b>Cross-Hospital Generalization (165 Subjects):</b><br/>"
        "Trained on 2 hospital cohorts and tested on an unseen 3rd site:<br/><br/>"
        "• <b>Galit Cohort (Ga, N=47):</b><br/>"
        "  1D CNN: 57.45% Acc, 0.657 AUC &rarr; <font color='#50E3C2'><b>BiLSTM: 61.70% Acc (+4.25%)</b></font><br/>"
        "• <b>Hausdorff Cohort (Ju, N=54):</b><br/>"
        "  1D CNN: 72.22% Acc, 0.797 AUC &rarr; <font color='#50E3C2'><b>BiLSTM: 77.78% Acc (+5.56%)</b></font><br/>"
        "• <b>Silberstein Cohort (Si, N=64):</b><br/>"
        "  1D CNN: 64.06% Acc, 0.776 AUC &rarr; <b>BiLSTM: 64.06% Acc</b><br/>"
        "• <b>Macro Average Across All 3 Cohorts:</b><br/>"
        "  1D CNN: 64.58% Acc, 0.744 AUC &rarr; <font color='#50E3C2'><b>BiLSTM: 67.85% Acc (+3.27% gain)</b></font><br/><br/>"
        "<b>SENSOR CORRUPTION RESILIENCE:</b><br/>"
        "• Tested under Gaussian sensor noise and channel dropout.<br/>"
        "• Calibrated Deep Ensembles maintain Expected Calibration Error <b>ECE &lt; 0.12</b>, "
        "whereas uncalibrated models degrade sharply (ECE &gt; 0.30).<br/><br/>"
        "<b>EVIDENTIAL ERROR DETECTION:</b><br/>"
        "Evidential uncertainty is directionally +20.16% higher on misclassified subjects (0.0590 vs 0.0491; p=0.14 trend)."
    )
    draw_card_with_text(c, 36, 36, 390, 408, "LEAVE-ONE-COHORT-OUT (LOCO)", lo_text, title_color=TEAL)

    # Right Column: Visual Dual-Plot
    draw_card(c, 442, 36, 482, 408)
    p_lp = Paragraph("<b>CROSS-HOSPITAL GENERALIZATION & CALIBRATION STABILITY</b>", style_card_title_blue)
    p_lp.wrap(450, 18)
    p_lp.drawOn(c, 458, 418)
    c.drawImage("presentation_assets/slide12_corruption_loco_dark.png", 452, 50, width=462, height=350, preserveAspectRatio=True)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 13: Interactive Demonstration & Live Walkthrough
    # =========================================================================
    draw_background(c)
    draw_header(c, "Interactive Demonstration & Live Walkthrough", "INTERACTIVE CLINICIAN DASHBOARD",
                "Streamlit application architecture (demo/app.py) and 90-second scenario walkthrough", 13)

    dm_text = (
        "<b>Four Integrated Application Views (demo/app.py):</b><br/>"
        "<b>1. Analyzer Dashboard:</b> Evaluates patient scenarios with ensemble gauges (M=5), "
        "evidential uncertainty meters, and split conformal prediction sets.<br/>"
        "<b>2. System Architecture View:</b> Interactive Mermaid dataflow visualizer.<br/>"
        "<b>3. Why Reliability? View:</b> Educational module demonstrating why probability averaging fails.<br/>"
        "<b>4. Experimental Evidence View:</b> Interactive LOCO benchmark tables and empirical corruption curves.<br/><br/>"
        "<b>LIVE 90-SECOND PRESENTATION SCRIPT:</b><br/>"
        "• <b>Action 1 (0:00-0:30): Scenario 1 (Strong Agreement)</b><br/>"
        "  Both Gait (89%) and Voice (79%) predict PD with matching singleton sets {PD}. Status: <b>ACCEPT</b>.<br/>"
        "• <b>Action 2 (0:30-1:00): Scenario 2 (Modality Conflict)</b><br/>"
        "  Gait predicts PD (85%) while Voice predicts HC (65%). System triggers <b>MODALITY_CONFLICT</b> "
        "rather than silently averaging to 60%.<br/>"
        "• <b>Action 3 (1:00-1:30): Transparency Panel</b><br/>"
        "  Inspect ensemble variance (&sigma;&sup2;=0.042), entropy (H=0.68), and conformal quantile (q_hat=0.8147)."
    )
    draw_card_with_text(c, 36, 36, 420, 408, "STREAMLIT DASHBOARD & LIVE SCRIPT", dm_text, title_color=TEAL)

    # Right Column: Visual Decision Flow
    draw_card(c, 472, 36, 452, 408)
    p_dp = Paragraph("<b>CROSS-MODAL RELIABILITY DECISION POLICY MATRIX</b>", style_card_title_amber)
    p_dp.wrap(420, 18)
    p_dp.drawOn(c, 488, 418)
    c.drawImage("presentation_assets/slide13_cross_modal_flow_dark.png", 482, 50, width=432, height=350, preserveAspectRatio=True)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 14: Limitations & Future Engineering Roadmap
    # =========================================================================
    draw_background(c)
    draw_header(c, "Project Limitations & Future Engineering Roadmap", "LIMITATIONS & TRANSLATIONAL ROADMAP",
                "Transparent scientific boundary definition and actionable translational engineering milestones", 14)

    lm_text = (
        "<b>1. Independent Multimodal Cohorts:</b><br/>"
        "Gait (PhysioNet) and Speech (MDVR-KCL) originate from separate clinical research studies. "
        "Cross-modal fusion is evaluated as simulated scenario evidence integration, not simultaneous "
        "patient-paired recording.<br/><br/>"
        "<b>2. Constrained Speech Sample Size:</b><br/>"
        "MDVR-KCL contains 37 total subjects, yielding an N = 6 held-out test split where a single sample "
        "misclassification causes a &plusmn;16.7% swing in reported accuracy.<br/><br/>"
        "<b>3. Statistical Power on Evidential Uncertainty:</b><br/>"
        "While evidential uncertainty is directionally +20.2% higher on incorrect predictions, the difference "
        "does not yet achieve formal p &lt; 0.05 significance (p = 0.14) due to limited test errors (N=8).<br/><br/>"
        "<b>4. Retrospective Offline Data:</b><br/>"
        "Models were developed and validated on retrospective benchmark repositories rather than deployed "
        "in prospective hospital clinical workflows."
    )
    draw_card_with_text(c, 36, 36, 430, 408, "CURRENT TECHNICAL LIMITATIONS", lm_text, title_color=AMBER, border_color=AMBER)

    rm_text = (
        "<b>1. Prospective Synchronized Cohorts:</b><br/>"
        "Validate the reliability engine on large prospective cohorts featuring simultaneous patient-paired "
        "gait and voice recordings (e.g., mPower smartphone study).<br/><br/>"
        "<b>2. Class-Conditional Conformal Prediction:</b><br/>"
        "Upgrade split conformal prediction to class-conditional coverage to enforce guaranteed 90% statistical "
        "coverage for PD and Healthy Control subjects independently.<br/><br/>"
        "<b>3. Wearable Sensor Integration:</b><br/>"
        "Extend biomechanical feature extractors from floor force plates to non-invasive tri-axial wearable "
        "accelerometers and gyroscopes for at-home continuous monitoring.<br/><br/>"
        "<b>4. Clinical Shadow-Mode Pilot:</b><br/>"
        "Deploy the Streamlit reliability dashboard in clinical shadow mode to assess physician triage speed, "
        "false-reassurance prevention, and diagnostic trust in hospital memory clinics."
    )
    draw_card_with_text(c, 482, 36, 442, 408, "ACTIONABLE ENGINEERING ROADMAP", rm_text, title_color=TEAL, border_color=TEAL)

    draw_footer(c)
    c.showPage()

    # =========================================================================
    # SLIDE 15: Conclusion & Technical Summary
    # =========================================================================
    draw_background(c)
    draw_header(c, "Conclusion & Technical Summary", "CONCLUSIONS & TAKEAWAYS",
                "Summary of validated engineering achievements, key findings, and reproducible deliverables", 15)

    # Top Row: 3 Bold KPI Callout Cards
    kpi_w = 282
    kpi_h = 105
    
    # KPI 1
    draw_card(c, 36, 339, kpi_w, kpi_h, border_color=TEAL)
    c.setFont('Helvetica-Bold', 24)
    c.setFillColor(TEAL)
    c.drawString(56, 408, "+14.3%")
    p_k1 = Paragraph("<b>Gait Sensitivity Gain</b><br/>BiLSTM (78.6%) vs 1D CNN Baseline (64.3%)",
                     ParagraphStyle('K1', parent=style_body, fontSize=8.5, leading=12.0, textColor=TEXT_MAIN))
    p_k1.wrap(kpi_w - 40, 40)
    p_k1.drawOn(c, 56, 355)

    # KPI 2
    draw_card(c, 339, 339, kpi_w, kpi_h, border_color=BLUE)
    c.setFont('Helvetica-Bold', 24)
    c.setFillColor(BLUE)
    c.drawString(359, 408, "67.85%")
    p_k2 = Paragraph("<b>LOCO Macro Generalization</b><br/>Outperforms 1D CNN (64.58%) across 3 external sites",
                     ParagraphStyle('K2', parent=style_body, fontSize=8.5, leading=12.0, textColor=TEXT_MAIN))
    p_k2.wrap(kpi_w - 40, 40)
    p_k2.drawOn(c, 359, 355)

    # KPI 3
    draw_card(c, 642, 339, kpi_w, kpi_h, border_color=AMBER)
    c.setFont('Helvetica-Bold', 24)
    c.setFillColor(AMBER)
    c.drawString(662, 408, "90.0%")
    p_k3 = Paragraph("<b>Guaranteed Statistical Coverage</b><br/>Split Conformal Prediction Sets (1 - &alpha; = 0.90)",
                     ParagraphStyle('K3', parent=style_body, fontSize=8.5, leading=12.0, textColor=TEXT_MAIN))
    p_k3.wrap(kpi_w - 40, 40)
    p_k3.drawOn(c, 662, 355)

    # Middle Card: Core Takeaways
    takeaway_text = (
        "<b>• Temporal Sequence Modeling is Essential for Gait:</b> Adding bidirectional LSTM recurrent layers "
        "to 1D convolutions decisively improves Parkinsonian gait detection, boosting sensitivity by +14.3% (78.57% vs 64.29%) "
        "and ROC-AUC from 0.6818 to 0.7532 on identical held-out test subjects.<br/><br/>"
        "<b>• Safety Over Blind Automation:</b> Machine learning models in healthcare must not operate as silent black boxes. "
        "Dirichlet evidential uncertainty heads, Deep Ensembles (M=5), and Split Conformal Prediction sets provide clinicians "
        "with actionable trust bounds rather than uncalibrated forced guesses.<br/><br/>"
        "<b>• Conflict-Aware Multimodal Fusion Prevents Silent Failures:</b> Rejecting naive probability averaging in favor of an "
        "explicit MODALITY_CONFLICT policy prevents false reassurance when sensory modalities disagree.<br/><br/>"
        "<b>• Reproducible Engineering Deliverables:</b> Complete verified codebase with clean 1D CNN baselines, automated unit tests, "
        "serialized checkpoints, and an interactive Streamlit demonstration dashboard."
    )
    draw_card_with_text(c, 36, 118, 888, 206, "CORE SCIENTIFIC & ENGINEERING TAKEAWAYS", takeaway_text, title_color=TEAL)

    # Bottom Banner: Q&A Invitation
    draw_card(c, 36, 36, 888, 70, border_color=TEAL)
    p_qa = Paragraph("<b>THANK YOU — QUESTIONS & TECHNICAL DISCUSSION</b>",
                     ParagraphStyle('QAHead', parent=style_title, fontSize=13, leading=16, textColor=TEAL, alignment=1))
    p_qa.wrap(850, 20)
    p_qa.drawOn(c, 56, 74)
    
    p_qa_sub = Paragraph("Reliability-Aware Parkinson's Disease Detection  |  Repository: d:\\projects\\parkinson-s-disease-detection",
                         ParagraphStyle('QASub', parent=style_sub, fontSize=9, leading=12, alignment=1))
    p_qa_sub.wrap(850, 15)
    p_qa_sub.drawOn(c, 56, 48)

    draw_footer(c)
    c.showPage()

    # Save PDF
    c.save()
    print(f"Presentation PDF successfully built and saved to {pdf_filename}!")

if __name__ == '__main__':
    render_deck()
