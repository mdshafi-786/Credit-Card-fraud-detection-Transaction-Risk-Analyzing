"""
FraudShield AI — Streamlit Application
Credit Card Fraud Detection & Transaction Risk Intelligence
"""

import streamlit as st
import json
import uuid
import sqlite3
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
import io
import os

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Paths & Constants
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "backend" / "transactions.db"
EDA_DIR = BASE_DIR / "notebooks" / "eda_outputs"
FRAUD_PROB_THRESHOLD = 0.50
RISK_SCORE_THRESHOLD = 60.0

CITIES = ["Bengaluru", "Mumbai", "Hyderabad", "Jaipur", "Ahmedabad",
          "Kolkata", "Chennai", "Visakhapatnam", "Pune", "Delhi"]
MERCHANT_CATEGORIES = ["Fuel", "Online Shopping", "Healthcare", "Restaurants",
                       "Clothing", "Travel", "Jewelry", "Grocery", "Electronics", "Entertainment"]
DEVICE_TYPES = ["Mobile", "Laptop", "Tablet", "POS Terminal"]
PAYMENT_METHODS = ["Online", "Contactless", "Chip & PIN", "Mobile Wallet"]
RISK_PROFILES = ["Low", "Medium", "High"]

# ─────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Custom CSS — Spatial Computing / visionOS UI Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ───── Import Fonts: Inter, Outfit, and JetBrains Mono ───── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ───── Spatial Root Variables ───── */
    :root {
        --spatial-bg: #030712;
        --spatial-card: rgba(15, 23, 42, 0.55);
        --spatial-card-hover: rgba(22, 33, 62, 0.75);
        --spatial-glass-top: rgba(255, 255, 255, 0.22);
        --spatial-glass-border: rgba(255, 255, 255, 0.12);
        --spatial-glow-indigo: rgba(99, 102, 241, 0.45);
        --spatial-glow-cyan: rgba(34, 211, 238, 0.45);
        --spatial-glow-rose: rgba(244, 63, 94, 0.45);
        --spatial-glow-emerald: rgba(16, 185, 129, 0.45);
        --accent-indigo: #6366f1;
        --accent-cyan: #22d3ee;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --accent-violet: #8b5cf6;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-display: 'Outfit', 'Inter', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* ───── Global Spatial Canvas ───── */
    .stApp {
        background-color: var(--spatial-bg) !important;
        background: radial-gradient(circle at 50% -10%, #111836 0%, #060919 45%, #030712 100%) !important;
        color: var(--text-primary);
        font-family: var(--font-main);
        -webkit-font-smoothing: antialiased;
        overflow-x: hidden;
    }

    /* Background Spatial Mesh & Orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: -150px;
        right: -100px;
        width: 700px;
        height: 700px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.18) 0%, rgba(139, 92, 246, 0.08) 45%, transparent 70%);
        border-radius: 50%;
        filter: blur(90px);
        pointer-events: none;
        z-index: 0;
        animation: floatOrb 22s ease-in-out infinite alternate;
    }

    .stApp::after {
        content: '';
        position: fixed;
        bottom: -150px;
        left: 5%;
        width: 650px;
        height: 650px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.14) 0%, rgba(99, 102, 241, 0.06) 50%, transparent 70%);
        border-radius: 50%;
        filter: blur(100px);
        pointer-events: none;
        z-index: 0;
        animation: floatOrb2 26s ease-in-out infinite alternate;
    }

    @keyframes floatOrb {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(40px, 30px) scale(1.08); }
        100% { transform: translate(-30px, 60px) scale(0.96); }
    }

    @keyframes floatOrb2 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-40px, -40px) scale(1.1); }
        100% { transform: translate(30px, -20px) scale(0.92); }
    }

    /* ───── Floating Spatial Sidebar (visionOS Dock) ───── */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 30, 0.65) !important;
        backdrop-filter: blur(36px) saturate(210%) !important;
        -webkit-backdrop-filter: blur(36px) saturate(210%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 12px 0 45px rgba(0, 0, 0, 0.55) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f1f5f9;
        font-family: var(--font-display);
    }

    /* VisionOS Navigation Segmented Controls */
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 8px;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-top: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        padding: 11px 18px;
        margin-bottom: 7px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: pointer;
        backdrop-filter: blur(16px);
        display: flex;
        align-items: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateX(4px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.2);
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.28), rgba(139, 92, 246, 0.18)) !important;
        border: 1px solid rgba(99, 102, 241, 0.55) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.35) !important;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.32), inset 0 1px 1px rgba(255, 255, 255, 0.28) !important;
        color: #ffffff !important;
        font-weight: 700;
        transform: scale(1.02);
    }

    /* ───── Spatial Floating HUD Top Bar ───── */
    .spatial-hud-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(28px) saturate(200%);
        -webkit-backdrop-filter: blur(28px) saturate(200%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 40px;
        padding: 8px 24px;
        margin-bottom: 24px;
        box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.15);
    }

    .spatial-hud-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .spatial-pulse-indicator {
        position: relative;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 12px #10b981;
    }

    .spatial-pulse-indicator::after {
        content: '';
        position: absolute;
        inset: -4px;
        border-radius: 50%;
        border: 1.5px solid #10b981;
        animation: pulseRing 1.8s cubic-bezier(0.24, 0, 0.38, 1) infinite;
    }

    @keyframes pulseRing {
        0% { transform: scale(0.7); opacity: 1; }
        100% { transform: scale(2.2); opacity: 0; }
    }

    .spatial-hud-title {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        background: linear-gradient(135deg, #a5b4fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: var(--font-mono);
    }

    .spatial-hud-right {
        display: flex;
        align-items: center;
        gap: 18px;
        font-size: 0.75rem;
        color: #94a3b8;
        font-family: var(--font-mono);
    }

    .spatial-hud-pill {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 4px 12px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* ───── Spatial Hero Banner (Window Floating in Space) ───── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.65) 0%, rgba(49, 46, 129, 0.45) 50%, rgba(30, 58, 138, 0.35) 100%) !important;
        backdrop-filter: blur(34px) saturate(210%) !important;
        -webkit-backdrop-filter: blur(34px) saturate(210%) !important;
        border-radius: 28px !important;
        padding: 38px 46px !important;
        margin-bottom: 30px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.32) !important;
        box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.65), 0 0 35px rgba(99, 102, 241, 0.12), inset 0 1px 1px rgba(255, 255, 255, 0.22) !important;
        position: relative;
        overflow: hidden;
    }

    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -15%;
        width: 550px;
        height: 550px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(34, 211, 238, 0.1) 40%, transparent 70%);
        border-radius: 50%;
        filter: blur(40px);
    }

    .hero-banner::after {
        content: 'SPATIAL COMPUTE NODE';
        position: absolute;
        top: 24px;
        right: 36px;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        color: rgba(165, 180, 252, 0.6);
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 5px 14px;
        border-radius: 30px;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        font-family: var(--font-display);
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 50%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        position: relative;
        letter-spacing: -0.02em;
        filter: drop-shadow(0 4px 20px rgba(99, 102, 241, 0.35));
    }

    .hero-subtitle {
        color: #c7d2fe;
        font-size: 1.05rem;
        font-weight: 400;
        position: relative;
        max-width: 85%;
        line-height: 1.6;
    }

    /* ───── Spatial KPI Cards (Floating Holo-Tiles) ───── */
    .kpi-card {
        background: rgba(15, 23, 42, 0.55) !important;
        backdrop-filter: blur(28px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(200%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.28) !important;
        border-radius: 24px !important;
        padding: 22px 18px !important;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.14) !important;
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 20%;
        right: 20%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    }

    .kpi-card:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
        border-top-color: rgba(255, 255, 255, 0.5) !important;
        transform: translateY(-7px) scale(1.025) !important;
        box-shadow: 0 24px 50px -10px rgba(99, 102, 241, 0.3), 0 0 25px rgba(99, 102, 241, 0.18), inset 0 1px 2px rgba(255, 255, 255, 0.35) !important;
    }

    .kpi-icon {
        font-size: 1.8rem;
        margin-bottom: 6px;
        filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.4));
    }

    .kpi-value {
        font-size: 2.15rem;
        font-weight: 800;
        font-family: var(--font-display);
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin: 6px 0 4px;
        letter-spacing: -0.02em;
        filter: drop-shadow(0 4px 16px rgba(99, 102, 241, 0.3));
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: var(--font-mono);
    }

    /* ───── Spatial Form Inputs & Dropdowns ───── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > input,
    div[data-baseweb="input"] input,
    .stTextInput input,
    .stNumberInput input {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 16px !important;
        color: #f8fafc !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.35) !important;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] input:focus,
    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.35), 0 0 24px rgba(99, 102, 241, 0.25) !important;
        background: rgba(20, 30, 55, 0.8) !important;
    }

    /* ───── Spatial Glass Buttons (visionOS Capsules) ───── */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-family: var(--font-display) !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        border-radius: 18px !important;
        padding: 12px 26px !important;
        letter-spacing: 0.03em !important;
        box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.5), inset 0 1px 2px rgba(255, 255, 255, 0.45) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 18px 40px -5px rgba(99, 102, 241, 0.7), 0 0 30px rgba(6, 182, 212, 0.45), inset 0 1px 2px rgba(255, 255, 255, 0.6) !important;
    }

    button[kind="secondary"], .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-family: var(--font-display) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.28) !important;
        border-radius: 18px !important;
        backdrop-filter: blur(20px) !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.18) !important;
    }

    button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.14) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 28px rgba(99, 102, 241, 0.25) !important;
        color: #ffffff !important;
    }

    /* ───── Spatial Expanders ───── */
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.1) !important;
        overflow: hidden;
    }

    /* ───── Status Badges with Pulsing Spatial Aura ───── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 24px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.06em;
        font-family: var(--font-display);
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }

    .badge-fraud {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.22), rgba(244, 63, 94, 0.28));
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.55);
        border-top: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 0 28px rgba(239, 68, 68, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.2);
        animation: spatialAuraRed 2s infinite alternate;
    }

    .badge-genuine {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.22), rgba(52, 211, 153, 0.28));
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.55);
        border-top: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 0 28px rgba(16, 185, 129, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.2);
        animation: spatialAuraGreen 2s infinite alternate;
    }

    .badge-suspicious {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.22), rgba(251, 191, 36, 0.28));
        color: #fde68a;
        border: 1px solid rgba(245, 158, 11, 0.55);
        border-top: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 0 28px rgba(245, 158, 11, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.2);
        animation: spatialAuraAmber 2s infinite alternate;
    }

    @keyframes spatialAuraRed {
        0% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
        100% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.65); }
    }

    @keyframes spatialAuraGreen {
        0% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.3); }
        100% { box-shadow: 0 0 35px rgba(16, 185, 129, 0.65); }
    }

    @keyframes spatialAuraAmber {
        0% { box-shadow: 0 0 15px rgba(245, 158, 11, 0.3); }
        100% { box-shadow: 0 0 35px rgba(245, 158, 11, 0.65); }
    }

    /* ───── Spatial Explainability Cards ───── */
    .reason-card {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-left: 4px solid #6366f1 !important;
        border-radius: 16px !important;
        padding: 15px 20px !important;
        margin-bottom: 10px !important;
        color: #e2e8f0 !important;
        font-size: 0.92rem !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.12) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .reason-card:hover {
        border-color: rgba(99, 102, 241, 0.45) !important;
        border-left: 4px solid #22d3ee !important;
        transform: translateX(5px) scale(1.01) !important;
        background: rgba(22, 33, 62, 0.8) !important;
        box-shadow: 0 14px 32px rgba(99, 102, 241, 0.22) !important;
    }

    /* ───── Section Headers with Glowing Accents ───── */
    .section-header {
        font-size: 1.45rem;
        font-weight: 700;
        font-family: var(--font-display);
        color: #f8fafc;
        margin-top: 14px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.01em;
    }

    /* ───── Spatial Result Cockpit Frame ───── */
    .spatial-glass-cockpit {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-top: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 26px;
        padding: 24px;
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        box-shadow: 0 20px 50px -10px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.16);
    }

    /* ───── Custom Spatial Scrollbars ───── */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(3, 7, 18, 0.6);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.18);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.6);
    }

    /* ───── Table Styling ───── */
    .dataframe, [data-testid="stDataFrame"] {
        border-radius: 18px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(15, 23, 42, 0.45) !important;
        backdrop-filter: blur(20px) !important;
    }

    /* ───── Spatial Footer ───── */
    .footer {
        text-align: center;
        padding: 28px;
        color: #64748b;
        font-size: 0.82rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 48px;
        background: rgba(15, 23, 42, 0.3);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        font-family: var(--font-mono);
    }

    /* Hide default Streamlit overhead */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Database Helpers (standalone, no backend imports)
# ─────────────────────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def db_exists():
    return DB_PATH.exists()


# ─────────────────────────────────────────────────────────────
# ML Engine (standalone, mirrors backend/app/ml/engine.py)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_engine():
    """Load all ML artifacts once and cache."""
    engine = {}
    try:
        engine["classifier"] = joblib.load(MODELS_DIR / "model_fraud_classifier.pkl")
        engine["regressor"] = joblib.load(MODELS_DIR / "model_risk_score.pkl")
        engine["scaler"] = joblib.load(MODELS_DIR / "scaler.pkl")
        engine["encoders"] = joblib.load(MODELS_DIR / "encoders.pkl")
        with open(MODELS_DIR / "feature_columns.json", "r") as f:
            engine["feature_columns"] = json.load(f)
        with open(MODELS_DIR / "feature_importances.json", "r") as f:
            engine["feature_importances"] = json.load(f)
        with open(MODELS_DIR / "threshold_config.json", "r") as f:
            engine["threshold_config"] = json.load(f)
        with open(MODELS_DIR / "metrics_summary.json", "r") as f:
            engine["metrics_summary"] = json.load(f)
        engine["loaded"] = True
    except Exception as e:
        engine["loaded"] = False
        engine["error"] = str(e)
    return engine


def prepare_features(engine, txn_dict):
    """Prepare feature vector from transaction dict."""
    dt = pd.to_datetime(txn_dict["transaction_datetime"])
    hour = dt.hour
    day_of_week = dt.dayofweek
    is_night = 1 if 0 <= hour <= 5 else 0

    home_loc = txn_dict["customer_home_location"]
    txn_loc = txn_dict["transaction_location"]
    inferred_mismatch = "Yes" if home_loc != txn_loc else "No"
    location_mismatch = txn_dict.get("location_mismatch") or inferred_mismatch
    loc_mismatch_bin = 1 if location_mismatch == "Yes" else 0

    distance = txn_dict.get("distance_from_home_km")
    if distance is None:
        distance = 0.0 if home_loc == txn_loc else 350.0

    avg_amt = txn_dict.get("customer_avg_transaction_amount")
    if avg_amt is None or avg_amt <= 0:
        avg_amt = float(txn_dict["transaction_amount"])

    amt = float(txn_dict["transaction_amount"])
    amount_vs_avg_ratio = amt / (avg_amt + 1e-5)

    tx_24h = int(txn_dict.get("transactions_last_24h") or 1)
    fails_24h = int(txn_dict.get("failed_attempts_last_24h") or 0)
    velocity_score = tx_24h + fails_24h

    risk_profile = txn_dict.get("customer_risk_profile", "Medium")
    risk_map = engine["encoders"].get("risk_map", {"Low": 0, "Medium": 1, "High": 2})
    risk_ord = risk_map.get(risk_profile, 1)

    intl_bin = 1 if txn_dict.get("international_transaction") == "Yes" else 0

    row = {col: 0.0 for col in engine["feature_columns"]}
    row["Transaction_Amount"] = amt
    row["Transaction_Hour"] = float(hour)
    row["Customer_Age"] = float(txn_dict.get("customer_age", 35))
    row["Customer_Income_Annual"] = float(txn_dict.get("customer_income_annual", 60000))
    row["Customer_Tenure_Years"] = float(txn_dict.get("customer_tenure_years", 3.0))
    row["Transactions_Last_30_Days"] = float(txn_dict.get("transactions_last_30_days", 15))
    row["Customer_Avg_Transaction_Amount"] = float(avg_amt)
    row["Distance_From_Home_km"] = float(distance)
    row["Transactions_Last_24h"] = float(tx_24h)
    row["Failed_Attempts_Last_24h"] = float(fails_24h)
    row["day_of_week"] = float(day_of_week)
    row["is_night_transaction"] = float(is_night)
    row["amount_vs_avg_ratio"] = float(amount_vs_avg_ratio)
    row["velocity_score"] = float(velocity_score)
    row["Customer_Risk_Profile_Ord"] = float(risk_ord)
    row["International_Transaction_Bin"] = float(intl_bin)
    row["Location_Mismatch_Bin"] = float(loc_mismatch_bin)

    one_hot_mappings = [
        (txn_dict.get("merchant_category", "Online Shopping"), "Merchant_Category_"),
        (txn_dict.get("device_type", "Mobile"), "Device_Type_"),
        (txn_dict.get("payment_method", "Online"), "Payment_Method_"),
        (home_loc, "Customer_Home_Location_"),
        (txn_loc, "Transaction_Location_")
    ]
    for val, prefix in one_hot_mappings:
        col_name = f"{prefix}{val}"
        if col_name in row:
            row[col_name] = 1.0

    df_features = pd.DataFrame([row])[engine["feature_columns"]]
    cols_to_scale = engine["encoders"]["numeric_cols_to_scale"]
    df_scaled = df_features.copy()
    df_scaled[cols_to_scale] = engine["scaler"].transform(df_features[cols_to_scale])

    augmented_info = {
        "hour": hour, "day_of_week": day_of_week, "is_night": is_night,
        "location_mismatch": location_mismatch, "loc_mismatch_bin": loc_mismatch_bin,
        "distance": distance, "avg_amt": avg_amt,
        "amount_vs_avg_ratio": amount_vs_avg_ratio, "velocity_score": velocity_score,
        "tx_24h": tx_24h, "fails_24h": fails_24h
    }
    return df_scaled, augmented_info


def generate_reasons(txn_dict, aug, prob, score):
    """Generate explainability reasons."""
    reasons = []
    if aug["amount_vs_avg_ratio"] >= 2.5:
        reasons.append(f"🔺 Spending Surge: Transaction amount (₹{txn_dict['transaction_amount']:,.2f}) is {aug['amount_vs_avg_ratio']:.1f}x higher than customer average (₹{aug['avg_amt']:,.2f})")
    elif aug["amount_vs_avg_ratio"] >= 1.5:
        reasons.append(f"📈 Elevated Ticket Size: Amount is {aug['amount_vs_avg_ratio']:.1f}x higher than typical ticket size")

    if aug["fails_24h"] >= 2:
        reasons.append(f"🔐 Authentication Warnings: {aug['fails_24h']} consecutive failed PIN/password attempts in the last 24 hours")
    elif aug["fails_24h"] == 1:
        reasons.append("🔐 Authentication Warning: 1 failed transaction attempt in the last 24 hours")

    if aug["loc_mismatch_bin"] == 1:
        reasons.append(f"📍 Geo-Spatial Discrepancy: Transaction in {txn_dict['transaction_location']}, but customer home is {txn_dict['customer_home_location']} ({aug['distance']:.0f} km away)")

    if aug["is_night"] == 1:
        reasons.append("🌙 Unusual Timing: Transaction during high-risk late night hours (12:00 AM - 05:59 AM)")

    if aug["velocity_score"] >= 4:
        reasons.append(f"⚡ High Velocity: {aug['velocity_score']} cumulative attempts & transactions within 24 hours")

    if txn_dict.get("international_transaction") == "Yes":
        reasons.append("🌍 Cross-Border Risk: International cross-border payment detected")

    if txn_dict.get("customer_risk_profile") == "High":
        reasons.append("⚠️ Customer Profile: Account marked under High Risk tier")

    if not reasons:
        if score < 40.0 and prob < 0.30:
            reasons.append("✅ Low Risk: Activity aligns with historical customer baseline and verified geographical profile")
        else:
            reasons.append("📊 Statistical Variance: Deviations detected across multiple multidimensional risk factors")

    return reasons[:4]


def predict_one(engine, txn_dict):
    """Run inference on a single transaction."""
    df_scaled, aug = prepare_features(engine, txn_dict)

    fraud_prob = float(engine["classifier"].predict_proba(df_scaled)[0, 1])
    fraud_threshold = engine["threshold_config"].get("fraud_probability_threshold", FRAUD_PROB_THRESHOLD)
    fraud_status = "Fraud" if fraud_prob >= fraud_threshold else "Genuine"

    risk_score_raw = float(engine["regressor"].predict(df_scaled)[0])
    risk_score = float(np.clip(risk_score_raw, 0.0, 100.0))

    susp_threshold = engine["threshold_config"].get("risk_score_suspicious_threshold", RISK_SCORE_THRESHOLD)
    rule_hit = (aug["loc_mismatch_bin"] == 1 and aug["distance"] > 150.0 and aug["fails_24h"] >= 2)
    is_suspicious = bool(risk_score >= susp_threshold or fraud_prob >= fraud_threshold or rule_hit)

    reasons = generate_reasons(txn_dict, aug, fraud_prob, risk_score)

    return {
        "fraud_status": fraud_status,
        "fraud_probability": round(fraud_prob, 4),
        "risk_score": round(risk_score, 2),
        "is_suspicious": is_suspicious,
        "top_reasons": reasons,
        "augmented_info": aug
    }


# ─────────────────────────────────────────────────────────────
# Dashboard Data Queries
# ─────────────────────────────────────────────────────────────
def get_dashboard_summary(filters=None):
    """Get aggregated dashboard summary from SQLite."""
    if not db_exists():
        return None

    conditions = []
    params = []
    if filters:
        if filters.get("merchant_category"):
            conditions.append("merchant_category = ?")
            params.append(filters["merchant_category"])
        if filters.get("location"):
            conditions.append("transaction_location = ?")
            params.append(filters["location"])
        if filters.get("device_type"):
            conditions.append("device_type = ?")
            params.append(filters["device_type"])
        if filters.get("risk_profile"):
            conditions.append("customer_risk_profile = ?")
            params.append(filters["risk_profile"])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count,
                SUM(CASE WHEN fraud_status = 'Genuine' THEN 1 ELSE 0 END) as genuine_count,
                AVG(transaction_amount) as avg_amount,
                AVG(fraud_risk_score) as avg_risk_score,
                SUM(CASE WHEN is_suspicious = 1 THEN 1 ELSE 0 END) as susp_count
            FROM transactions {where_clause}
        """, params)
        agg = cursor.fetchone()
        total = agg["total"] or 0
        fraud_count = agg["fraud_count"] or 0
        genuine_count = agg["genuine_count"] or 0
        avg_amount = float(agg["avg_amount"] or 0.0)
        avg_risk_score = float(agg["avg_risk_score"] or 0.0)
        susp_count = agg["susp_count"] or 0

        # Breakdown by Category
        cursor.execute(f"""
            SELECT merchant_category as category, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count,
                   AVG(fraud_risk_score) as avg_risk_score
            FROM transactions {where_clause}
            GROUP BY merchant_category ORDER BY count DESC
        """, params)
        cat_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in cat_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Breakdown by Location Mismatch
        cursor.execute(f"""
            SELECT location_mismatch, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY location_mismatch
        """, params)
        loc_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in loc_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Breakdown by Device
        cursor.execute(f"""
            SELECT device_type, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY device_type
        """, params)
        dev_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in dev_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Breakdown by Payment Method
        cursor.execute(f"""
            SELECT payment_method, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY payment_method
        """, params)
        pay_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in pay_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Breakdown by Hour
        cursor.execute(f"""
            SELECT transaction_hour as hour, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY transaction_hour ORDER BY transaction_hour ASC
        """, params)
        hour_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in hour_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Breakdown by Risk Profile
        cursor.execute(f"""
            SELECT customer_risk_profile as risk_profile, COUNT(*) as count,
                   SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY customer_risk_profile
        """, params)
        profile_breakdown = [dict(r) for r in cursor.fetchall()]
        for c in profile_breakdown:
            c["fraud_rate"] = round(c["fraud_count"] / c["count"] * 100, 2) if c["count"] > 0 else 0.0

        # Risk score distribution
        cursor.execute(f"""
            SELECT 
                CASE 
                    WHEN fraud_risk_score < 20 THEN '0-20'
                    WHEN fraud_risk_score < 40 THEN '20-40'
                    WHEN fraud_risk_score < 60 THEN '40-60'
                    WHEN fraud_risk_score < 80 THEN '60-80'
                    ELSE '80-100'
                END as bin,
                COUNT(*) as count,
                SUM(CASE WHEN fraud_status = 'Fraud' THEN 1 ELSE 0 END) as fraud_count
            FROM transactions {where_clause}
            GROUP BY bin ORDER BY bin ASC
        """, params)
        risk_dist = [dict(r) for r in cursor.fetchall()]

    fraud_rate_pct = round((fraud_count / total * 100), 2) if total > 0 else 0.0
    susp_rate_pct = round((susp_count / total * 100), 2) if total > 0 else 0.0

    return {
        "total_transactions": total,
        "fraud_transactions": fraud_count,
        "genuine_transactions": genuine_count,
        "fraud_rate_pct": fraud_rate_pct,
        "average_transaction_amount": round(avg_amount, 2),
        "average_fraud_risk_score": round(avg_risk_score, 2),
        "suspicious_transactions": susp_count,
        "suspicious_rate_pct": susp_rate_pct,
        "breakdown_by_category": cat_breakdown,
        "breakdown_by_location_mismatch": loc_breakdown,
        "breakdown_by_device": dev_breakdown,
        "breakdown_by_payment_method": pay_breakdown,
        "breakdown_by_hour": hour_breakdown,
        "breakdown_by_risk_profile": profile_breakdown,
        "risk_score_distribution": risk_dist
    }


def get_transactions_df(page=1, page_size=50, filters=None):
    """Fetch paginated transaction data as DataFrame."""
    if not db_exists():
        return pd.DataFrame(), 0

    conditions = []
    params = []
    if filters:
        if filters.get("search"):
            conditions.append("(transaction_id LIKE ? OR transaction_location LIKE ?)")
            term = f"%{filters['search']}%"
            params.extend([term, term])
        if filters.get("fraud_status"):
            conditions.append("fraud_status = ?")
            params.append(filters["fraud_status"])
        if filters.get("merchant_category"):
            conditions.append("merchant_category = ?")
            params.append(filters["merchant_category"])
        if filters.get("is_suspicious") is not None:
            conditions.append("is_suspicious = ?")
            params.append(1 if filters["is_suspicious"] else 0)
        if filters.get("risk_level"):
            rl = filters["risk_level"]
            if rl == "Low":
                conditions.append("fraud_risk_score < 40.0")
            elif rl == "Medium":
                conditions.append("fraud_risk_score >= 40.0 AND fraud_risk_score < 70.0")
            elif rl == "High":
                conditions.append("fraud_risk_score >= 70.0")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM transactions {where_clause}", params)
        total = cursor.fetchone()[0]

        offset = (page - 1) * page_size
        cursor.execute(f"""
            SELECT * FROM transactions {where_clause}
            ORDER BY transaction_datetime DESC
            LIMIT ? OFFSET ?
        """, params + [page_size, offset])
        rows = cursor.fetchall()
        if rows:
            df = pd.DataFrame([dict(r) for r in rows])
        else:
            df = pd.DataFrame()

    return df, total


# ─────────────────────────────────────────────────────────────
# Plotly Theme Helpers
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit, Inter, sans-serif", color="#cbd5e1"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
)

COLOR_PALETTE = ["#6366f1", "#22d3ee", "#f43f5e", "#10b981", "#f59e0b",
                  "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#3b82f6"]


# ─────────────────────────────────────────────────────────────
# Render Risk Score Gauge (Spatial Holographic Plotly)
# ─────────────────────────────────────────────────────────────
def render_risk_gauge(score, fraud_prob, fraud_status, is_suspicious):
    """Render a futuristic Spatial HUD risk score gauge."""
    if score < 40:
        gauge_color = "#10b981"
        bar_bg = "rgba(16, 185, 129, 0.15)"
    elif score < 70:
        gauge_color = "#f59e0b"
        bar_bg = "rgba(245, 158, 11, 0.15)"
    else:
        gauge_color = "#f43f5e"
        bar_bg = "rgba(244, 63, 94, 0.2)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(
            font=dict(size=52, color=gauge_color, family="Outfit, Inter"),
            suffix="/100",
            valueformat=".1f"
        ),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#475569", tickwidth=1.5, tickfont=dict(color="#94a3b8", family="JetBrains Mono")),
            bar=dict(color=gauge_color, thickness=0.28),
            bgcolor=bar_bg,
            borderwidth=1,
            bordercolor="rgba(255, 255, 255, 0.12)",
            steps=[
                dict(range=[0, 40], color="rgba(16, 185, 129, 0.10)"),
                dict(range=[40, 70], color="rgba(245, 158, 11, 0.10)"),
                dict(range=[70, 100], color="rgba(244, 63, 94, 0.14)"),
            ],
            threshold=dict(
                line=dict(color="#ffffff", width=3),
                thickness=0.85,
                value=60
            ),
        ),
        title=dict(
            text="<span style='font-family: Outfit; font-weight: 700; letter-spacing: 0.1em; color: #cbd5e1;'>RISK INDEX TELEMETRY</span>",
            font=dict(size=13)
        ),
    ))
    fig.update_layout(
        height=300,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ['xaxis', 'yaxis']},
        margin=dict(l=30, r=30, t=65, b=25)
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Sidebar Navigation (VisionOS Floating Spatial Dock)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 18px 0 22px;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 22px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.35), rgba(6, 182, 212, 0.25)); border: 1px solid rgba(255, 255, 255, 0.25); box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35); font-size: 2.2rem; margin-bottom: 10px;">
            🛡️
        </div>
        <div style="font-size: 1.35rem; font-weight: 800; font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 60%, #818cf8 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: -0.01em;">FraudShield AI</div>
        <div style="font-size: 0.7rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px;">
            SPATIAL RISK COGNITION
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Transaction Check", "📦 Batch Upload",
         "📋 Transaction Explorer", "🧠 Model Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Model status telemetry pills
    engine = load_ml_engine()
    if engine.get("loaded"):
        st.markdown("""
        <div class="spatial-hud-pill" style="margin-bottom: 8px; justify-content: center; background: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.35); color: #6ee7b7; font-weight: 600;">
            <span style="font-size: 0.9rem;">✅</span> ML Engines Online
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="spatial-hud-pill" style="margin-bottom: 8px; justify-content: center; background: rgba(239, 68, 68, 0.12); border-color: rgba(239, 68, 68, 0.35); color: #fca5a5; font-weight: 600;">
            <span style="font-size: 0.9rem;">❌</span> ML Engine Error
        </div>
        """, unsafe_allow_html=True)

    if db_exists():
        with get_db() as conn:
            count = conn.cursor().execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        st.markdown(f"""
        <div class="spatial-hud-pill" style="justify-content: center; background: rgba(99, 102, 241, 0.12); border-color: rgba(99, 102, 241, 0.35); color: #c7d2fe; font-weight: 600;">
            <span style="font-size: 0.9rem;">💾</span> Data Lake: {count:,} txns
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="spatial-hud-pill" style="justify-content: center; background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.35); color: #fde68a; font-weight: 600;">
            <span style="font-size: 0.9rem;">⚠️</span> No database found
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 32px; padding: 12px; border-radius: 14px; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); text-align: center; color: #64748b; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;">
        SPATIAL UI v2.0 • STREAMLIT CORE<br>VISION COMPUTING MATRIX
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Top Spatial HUD Status Ribbon
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="spatial-hud-bar">
    <div class="spatial-hud-left">
        <div class="spatial-pulse-indicator"></div>
        <div class="spatial-hud-title">SPATIAL RISK COCKPIT • ACTIVE</div>
    </div>
    <div class="spatial-hud-right">
        <div class="spatial-hud-pill">⚡ DUAL ML CORE</div>
        <div class="spatial-hud-pill">🛡️ QUANTUM HEURISTICS</div>
        <div class="spatial-hud-pill">⏱️ 0.02ms INFERENCE</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 1: Dashboard
# ─────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Fraud Analytics Dashboard</div>
        <div class="hero-subtitle">
            Real-time intelligence across 25,000+ transactions • Dual ML architecture • 
            Logistic Regression + LightGBM + Behavioral Heuristics
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    with st.expander("🎛️ Global Filters", expanded=False):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            f_cat = st.selectbox("Merchant Category", ["All"] + MERCHANT_CATEGORIES, key="dash_cat")
        with fcol2:
            f_loc = st.selectbox("Location", ["All"] + CITIES, key="dash_loc")
        with fcol3:
            f_dev = st.selectbox("Device Type", ["All"] + DEVICE_TYPES, key="dash_dev")
        with fcol4:
            f_risk = st.selectbox("Risk Profile", ["All"] + RISK_PROFILES, key="dash_risk")

    filters = {}
    if f_cat != "All":
        filters["merchant_category"] = f_cat
    if f_loc != "All":
        filters["location"] = f_loc
    if f_dev != "All":
        filters["device_type"] = f_dev
    if f_risk != "All":
        filters["risk_profile"] = f_risk

    summary = get_dashboard_summary(filters or None)

    if summary:
        # KPI Row
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpis = [
            (k1, "📊", f"{summary['total_transactions']:,}", "Total Transactions"),
            (k2, "🚨", f"{summary['fraud_transactions']:,}", f"Fraud ({summary['fraud_rate_pct']}%)"),
            (k3, "✅", f"{summary['genuine_transactions']:,}", "Genuine"),
            (k4, "💰", f"₹{summary['average_transaction_amount']:,.0f}", "Avg Amount"),
            (k5, "🎯", f"{summary['average_fraud_risk_score']:.1f}", "Avg Risk Score"),
            (k6, "⚡", f"{summary['suspicious_transactions']:,}", f"Suspicious ({summary['suspicious_rate_pct']}%)"),
        ]
        for col, icon, value, label in kpis:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 1
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-header">📊 Fraud Rate by Merchant Category</div>', unsafe_allow_html=True)
            cat_df = pd.DataFrame(summary["breakdown_by_category"])
            if not cat_df.empty:
                fig = px.bar(
                    cat_df, x="category", y="fraud_rate",
                    color="fraud_rate",
                    color_continuous_scale=["#6366f1", "#f43f5e"],
                    labels={"fraud_rate": "Fraud Rate (%)", "category": "Category"},
                    text="fraud_rate"
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">⏰ Hourly Fraud Rate Trend</div>', unsafe_allow_html=True)
            hour_df = pd.DataFrame(summary["breakdown_by_hour"])
            if not hour_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hour_df["hour"], y=hour_df["fraud_rate"],
                    mode="lines+markers",
                    line=dict(color="#22d3ee", width=3, shape="spline"),
                    marker=dict(size=7, color="#22d3ee", line=dict(color="#0e7490", width=1)),
                    fill="tozeroy",
                    fillcolor="rgba(34, 211, 238, 0.08)",
                    name="Fraud Rate %"
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT, height=400,
                    xaxis_title="Hour of Day",
                    yaxis_title="Fraud Rate (%)"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Charts Row 2
        col_left2, col_right2 = st.columns(2)

        with col_left2:
            st.markdown('<div class="section-header">📱 Fraud by Device Type</div>', unsafe_allow_html=True)
            dev_df = pd.DataFrame(summary["breakdown_by_device"])
            if not dev_df.empty:
                fig = px.bar(
                    dev_df, x="device_type", y=["count", "fraud_count"],
                    barmode="group",
                    color_discrete_sequence=["#6366f1", "#f43f5e"],
                    labels={"value": "Count", "device_type": "Device"}
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=380)
                st.plotly_chart(fig, use_container_width=True)

        with col_right2:
            st.markdown('<div class="section-header">🎯 Risk Score Distribution</div>', unsafe_allow_html=True)
            risk_df = pd.DataFrame(summary["risk_score_distribution"])
            if not risk_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=risk_df["bin"], y=risk_df["count"],
                    name="Total",
                    marker_color="#6366f1",
                    marker_line=dict(color="#818cf8", width=1)
                ))
                fig.add_trace(go.Bar(
                    x=risk_df["bin"], y=risk_df["fraud_count"],
                    name="Fraud",
                    marker_color="#f43f5e",
                    marker_line=dict(color="#fb7185", width=1)
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=380, barmode="overlay")
                fig.update_traces(opacity=0.8, selector=dict(name="Total"))
                st.plotly_chart(fig, use_container_width=True)

        # Charts Row 3
        col_left3, col_right3 = st.columns(2)

        with col_left3:
            st.markdown('<div class="section-header">📍 Location Mismatch Impact</div>', unsafe_allow_html=True)
            loc_df = pd.DataFrame(summary["breakdown_by_location_mismatch"])
            if not loc_df.empty:
                fig = px.pie(
                    loc_df, names="location_mismatch", values="fraud_count",
                    color_discrete_sequence=["#10b981", "#f43f5e"],
                    hole=0.5
                )
                fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ['xaxis', 'yaxis']}, height=380)
                fig.update_traces(textinfo="label+percent", textfont_size=13)
                st.plotly_chart(fig, use_container_width=True)

        with col_right3:
            st.markdown('<div class="section-header">👤 Customer Risk Profile vs Fraud</div>', unsafe_allow_html=True)
            prof_df = pd.DataFrame(summary["breakdown_by_risk_profile"])
            if not prof_df.empty:
                fig = px.bar(
                    prof_df, x="risk_profile", y="fraud_rate",
                    color="risk_profile",
                    color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                    labels={"fraud_rate": "Fraud Rate (%)", "risk_profile": "Profile"},
                    text="fraud_rate"
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Payment Method
        st.markdown('<div class="section-header">💳 Fraud Rate by Payment Method</div>', unsafe_allow_html=True)
        pay_df = pd.DataFrame(summary["breakdown_by_payment_method"])
        if not pay_df.empty:
            fig = px.bar(
                pay_df, x="payment_method", y="fraud_rate",
                color="fraud_rate",
                color_continuous_scale=["#8b5cf6", "#f43f5e"],
                labels={"fraud_rate": "Fraud Rate (%)", "payment_method": "Payment Method"},
                text="fraud_rate"
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No database available. Please seed the database first.")


# ─────────────────────────────────────────────────────────────
# PAGE 2: Transaction Check
# ─────────────────────────────────────────────────────────────
elif page == "🔍 Transaction Check":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔍 Real-Time Transaction Check</div>
        <div class="hero-subtitle">
            Score any transaction instantly using dual ML models • Get fraud probability, 
            risk score, and AI-driven explainability reasons
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Presets
    st.markdown('<div class="section-header">⚡ Quick Test Presets</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        if st.button("✅ Safe Transaction", use_container_width=True, type="secondary"):
            st.session_state["preset"] = {
                "amount": 2500.0, "category": "Grocery", "age": 32,
                "income": 750000, "risk": "Low", "home": "Mumbai",
                "txn_loc": "Mumbai", "device": "Mobile", "payment": "Online",
                "intl": "No", "fails": 0, "tx24h": 2
            }
    with pc2:
        if st.button("⚠️ Medium Risk", use_container_width=True, type="secondary"):
            st.session_state["preset"] = {
                "amount": 15000.0, "category": "Electronics", "age": 28,
                "income": 500000, "risk": "Medium", "home": "Delhi",
                "txn_loc": "Jaipur", "device": "Laptop", "payment": "Contactless",
                "intl": "No", "fails": 1, "tx24h": 4
            }
    with pc3:
        if st.button("🚨 High Risk / Fraud", use_container_width=True, type="primary"):
            st.session_state["preset"] = {
                "amount": 48000.0, "category": "Jewelry", "age": 45,
                "income": 400000, "risk": "High", "home": "Chennai",
                "txn_loc": "Delhi", "device": "Tablet", "payment": "Mobile Wallet",
                "intl": "Yes", "fails": 3, "tx24h": 7
            }

    preset = st.session_state.get("preset", {})

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form
    with st.form("txn_form", clear_on_submit=False):
        st.markdown('<div class="section-header">📝 Transaction Details</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            amount = st.number_input("💰 Transaction Amount (₹)", min_value=1.0,
                                     value=preset.get("amount", 5000.0), step=100.0)
        with c2:
            category = st.selectbox("🏪 Merchant Category", MERCHANT_CATEGORIES,
                                    index=MERCHANT_CATEGORIES.index(preset.get("category", "Online Shopping")))
        with c3:
            txn_date = st.date_input("📅 Transaction Date", value=datetime.now().date())
        with c4:
            txn_time = st.time_input("🕐 Transaction Time", value=datetime.now().time())

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            age = st.number_input("👤 Customer Age", min_value=18, max_value=100,
                                  value=preset.get("age", 35))
        with c6:
            income = st.number_input("💼 Annual Income (₹)", min_value=0,
                                     value=preset.get("income", 600000), step=50000)
        with c7:
            risk_profile = st.selectbox("⚠️ Risk Profile", RISK_PROFILES,
                                        index=RISK_PROFILES.index(preset.get("risk", "Medium")))
        with c8:
            home_loc = st.selectbox("🏠 Home City", CITIES,
                                    index=CITIES.index(preset.get("home", "Mumbai")))

        c9, c10, c11, c12 = st.columns(4)
        with c9:
            txn_loc = st.selectbox("📍 Transaction City", CITIES,
                                   index=CITIES.index(preset.get("txn_loc", "Mumbai")))
        with c10:
            device = st.selectbox("📱 Device Type", DEVICE_TYPES,
                                  index=DEVICE_TYPES.index(preset.get("device", "Mobile")))
        with c11:
            payment = st.selectbox("💳 Payment Method", PAYMENT_METHODS,
                                   index=PAYMENT_METHODS.index(preset.get("payment", "Online")))
        with c12:
            intl = st.selectbox("🌍 International?", ["No", "Yes"],
                                index=1 if preset.get("intl") == "Yes" else 0)

        # Advanced
        with st.expander("🔧 Advanced Attributes"):
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                tenure = st.number_input("Tenure (years)", min_value=0.0, value=3.0, step=0.5)
            with ac2:
                txns_30d = st.number_input("Txns Last 30 Days", min_value=0, value=15)
            with ac3:
                failed_attempts = st.number_input("Failed Attempts (24h)", min_value=0,
                                                   value=preset.get("fails", 0))
            with ac4:
                txns_24h = st.number_input("Txns Last 24h", min_value=0,
                                           value=preset.get("tx24h", 1))

        submitted = st.form_submit_button("🚀 Analyze Transaction", use_container_width=True, type="primary")

    if submitted and engine.get("loaded"):
        txn_datetime = datetime.combine(txn_date, txn_time).strftime("%Y-%m-%d %H:%M:%S")
        txn_dict = {
            "transaction_amount": amount,
            "merchant_category": category,
            "transaction_datetime": txn_datetime,
            "customer_age": age,
            "customer_income_annual": income,
            "customer_risk_profile": risk_profile,
            "customer_home_location": home_loc,
            "transaction_location": txn_loc,
            "device_type": device,
            "payment_method": payment,
            "international_transaction": intl,
            "customer_tenure_years": tenure,
            "transactions_last_30_days": txns_30d,
            "failed_attempts_last_24h": failed_attempts,
            "transactions_last_24h": txns_24h,
        }

        with st.spinner("🔄 Running ML inference..."):
            result = predict_one(engine, txn_dict)

        st.markdown("---")
        st.markdown('<div class="section-header">📊 Spatial Analysis Results</div>', unsafe_allow_html=True)

        # Results layout with spatial cockpit glass containers
        res_left, res_mid, res_right = st.columns([1.1, 0.9, 1.2])

        with res_left:
            st.markdown('<div class="spatial-glass-cockpit" style="height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center;">', unsafe_allow_html=True)
            gauge_fig = render_risk_gauge(
                result["risk_score"], result["fraud_probability"],
                result["fraud_status"], result["is_suspicious"]
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with res_mid:
            badge_class = "badge-fraud" if result["fraud_status"] == "Fraud" else "badge-genuine"
            st.markdown(f"""
            <div class="spatial-glass-cockpit" style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 24px 16px;">
                <div style="color: #94a3b8; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 12px;">PREDICTIVE DECISION</div>
                <div style="margin-bottom: 18px;">
                    <span class="status-badge {badge_class}">
                        {'🚨' if result['fraud_status'] == 'Fraud' else '✅'} {result['fraud_status']}
                    </span>
                </div>
                {f'''<div style="margin-bottom: 16px;">
                    <span class="status-badge badge-suspicious">⚡ SUSPICIOUS DRIFT</span>
                </div>''' if result["is_suspicious"] else ''}
                <div style="color: #94a3b8; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 6px;">FRAUD PROBABILITY</div>
                <div style="font-size: 2.8rem; font-weight: 800; font-family: 'Outfit', sans-serif; color: {'#f43f5e' if result['fraud_probability'] >= 0.5 else '#10b981'}; filter: drop-shadow(0 4px 16px {'rgba(244,63,94,0.45)' if result['fraud_probability'] >= 0.5 else 'rgba(16,185,129,0.45)'});">
                    {result['fraud_probability']*100:.1f}%
                </div>
                <div style="font-size: 0.72rem; color: #64748b; font-family: 'JetBrains Mono', monospace; margin-top: 6px;">
                    DECISION BOUNDARY: {FRAUD_PROB_THRESHOLD*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_right:
            st.markdown('<div class="spatial-glass-cockpit" style="height: 100%; padding: 22px 20px;">', unsafe_allow_html=True)
            st.markdown('<div style="color: #94a3b8; font-size: 0.78rem; font-family: \'JetBrains Mono\', monospace; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 14px;">🧠 AI EXPLAINABILITY DRIVERS</div>', unsafe_allow_html=True)
            for reason in result["top_reasons"]:
                st.markdown(f"""
                <div class="reason-card">
                    <div>{reason}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3: Batch Upload
# ─────────────────────────────────────────────────────────────
elif page == "📦 Batch Upload":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📦 Batch CSV Scoring</div>
        <div class="hero-subtitle">
            Upload a CSV file with multiple transactions for bulk fraud scoring • 
            Download results with risk scores and explainability
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample CSV Template
    with st.expander("📥 Download Sample CSV Template"):
        sample_data = {
            "transaction_amount": [2500, 15000, 48000],
            "merchant_category": ["Grocery", "Electronics", "Jewelry"],
            "transaction_datetime": ["2025-06-15 10:30:00", "2025-06-15 14:00:00", "2025-06-16 02:15:00"],
            "customer_age": [32, 28, 45],
            "customer_income_annual": [750000, 500000, 400000],
            "customer_risk_profile": ["Low", "Medium", "High"],
            "customer_home_location": ["Mumbai", "Delhi", "Chennai"],
            "transaction_location": ["Mumbai", "Jaipur", "Delhi"],
            "device_type": ["Mobile", "Laptop", "Tablet"],
            "payment_method": ["Online", "Contactless", "Mobile Wallet"],
            "international_transaction": ["No", "No", "Yes"],
            "failed_attempts_last_24h": [0, 1, 3],
            "transactions_last_24h": [2, 4, 7],
        }
        sample_df = pd.DataFrame(sample_data)
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Template CSV",
            csv_buffer.getvalue(),
            "fraud_check_template.csv",
            "text/csv",
            use_container_width=True
        )
        st.dataframe(sample_df, use_container_width=True)

    # File Upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV with transaction data for batch scoring"
    )

    if uploaded_file is not None and engine.get("loaded"):
        df_upload = pd.read_csv(uploaded_file)
        st.markdown(f"**Uploaded:** {len(df_upload)} transactions")

        if st.button("🚀 Score All Transactions", type="primary", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, row in df_upload.iterrows():
                txn_dict = row.to_dict()
                # Ensure required fields have defaults
                for field in ["device_type", "payment_method", "international_transaction"]:
                    if field not in txn_dict or pd.isna(txn_dict.get(field)):
                        if field == "device_type":
                            txn_dict[field] = "Mobile"
                        elif field == "payment_method":
                            txn_dict[field] = "Online"
                        elif field == "international_transaction":
                            txn_dict[field] = "No"

                for field in ["failed_attempts_last_24h", "transactions_last_24h"]:
                    if field not in txn_dict or pd.isna(txn_dict.get(field)):
                        txn_dict[field] = 0 if "failed" in field else 1

                try:
                    pred = predict_one(engine, txn_dict)
                    results.append({
                        "transaction_id": txn_dict.get("transaction_id", f"BATCH_{idx+1:04d}"),
                        "transaction_amount": txn_dict["transaction_amount"],
                        "merchant_category": txn_dict.get("merchant_category", ""),
                        "fraud_status": pred["fraud_status"],
                        "fraud_probability": pred["fraud_probability"],
                        "risk_score": pred["risk_score"],
                        "is_suspicious": pred["is_suspicious"],
                        "top_reason": pred["top_reasons"][0] if pred["top_reasons"] else "",
                    })
                except Exception as e:
                    results.append({
                        "transaction_id": f"BATCH_{idx+1:04d}",
                        "transaction_amount": txn_dict.get("transaction_amount", 0),
                        "merchant_category": txn_dict.get("merchant_category", ""),
                        "fraud_status": "ERROR",
                        "fraud_probability": 0,
                        "risk_score": 0,
                        "is_suspicious": False,
                        "top_reason": str(e),
                    })

                progress_bar.progress((idx + 1) / len(df_upload))
                status_text.text(f"Processing {idx+1}/{len(df_upload)}...")

            status_text.text("✅ Batch scoring complete!")
            results_df = pd.DataFrame(results)

            # Summary KPIs
            st.markdown("---")
            fraud_count = len(results_df[results_df["fraud_status"] == "Fraud"])
            susp_count = len(results_df[results_df["is_suspicious"] == True])
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Total Processed", len(results_df))
            with bc2:
                st.metric("🚨 Fraud Detected", fraud_count)
            with bc3:
                st.metric("⚡ Suspicious", susp_count)

            # Results Table
            st.markdown('<div class="section-header">📊 Scored Results</div>', unsafe_allow_html=True)
            st.dataframe(
                results_df.style.applymap(
                    lambda v: "background-color: rgba(239,68,68,0.15); color: #f87171;"
                    if v == "Fraud" else ("background-color: rgba(16,185,129,0.15); color: #34d399;" if v == "Genuine" else ""),
                    subset=["fraud_status"]
                ),
                use_container_width=True,
                height=400
            )

            # Download
            csv_out = io.StringIO()
            results_df.to_csv(csv_out, index=False)
            st.download_button(
                "⬇️ Download Scored CSV",
                csv_out.getvalue(),
                "batch_scored_results.csv",
                "text/csv",
                use_container_width=True,
                type="primary"
            )


# ─────────────────────────────────────────────────────────────
# PAGE 4: Transaction Explorer
# ─────────────────────────────────────────────────────────────
elif page == "📋 Transaction Explorer":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📋 Transaction Explorer</div>
        <div class="hero-subtitle">
            Search, filter, and explore all 25,000+ scored transactions in the database • 
            Drill-down into individual records
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not db_exists():
        st.warning("⚠️ Database not found. Please seed the database first.")
    else:
        # Filters
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        with fc1:
            search = st.text_input("🔍 Search ID/Location", "")
        with fc2:
            fs = st.selectbox("Fraud Status", ["All", "Fraud", "Genuine"], key="exp_fs")
        with fc3:
            mc = st.selectbox("Category", ["All"] + MERCHANT_CATEGORIES, key="exp_mc")
        with fc4:
            rl = st.selectbox("Risk Level", ["All", "Low", "Medium", "High"], key="exp_rl")
        with fc5:
            susp = st.selectbox("Suspicious", ["All", "Yes", "No"], key="exp_susp")

        filters = {}
        if search:
            filters["search"] = search
        if fs != "All":
            filters["fraud_status"] = fs
        if mc != "All":
            filters["merchant_category"] = mc
        if rl != "All":
            filters["risk_level"] = rl
        if susp != "All":
            filters["is_suspicious"] = susp == "Yes"

        # Pagination
        page_size = 50
        if "explorer_page" not in st.session_state:
            st.session_state["explorer_page"] = 1

        df_txns, total = get_transactions_df(
            page=st.session_state["explorer_page"],
            page_size=page_size,
            filters=filters or None
        )

        total_pages = max(1, (total + page_size - 1) // page_size)

        st.markdown(f"**Showing {len(df_txns)} of {total:,} transactions** | Page {st.session_state['explorer_page']} of {total_pages}")

        # Navigation
        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("⬅️ Previous", disabled=st.session_state["explorer_page"] <= 1, use_container_width=True):
                st.session_state["explorer_page"] -= 1
                st.rerun()
        with nav3:
            if st.button("➡️ Next", disabled=st.session_state["explorer_page"] >= total_pages, use_container_width=True):
                st.session_state["explorer_page"] += 1
                st.rerun()

        if not df_txns.empty:
            # Select columns to display
            display_cols = [
                "transaction_id", "transaction_amount", "merchant_category",
                "transaction_datetime", "transaction_location", "fraud_status",
                "fraud_probability", "fraud_risk_score", "is_suspicious",
                "customer_risk_profile", "device_type", "payment_method"
            ]
            available_cols = [c for c in display_cols if c in df_txns.columns]
            st.dataframe(
                df_txns[available_cols],
                use_container_width=True,
                height=500
            )

            # Detail View
            st.markdown("---")
            st.markdown('<div class="section-header">🔎 Transaction Detail View</div>', unsafe_allow_html=True)
            selected_id = st.selectbox(
                "Select Transaction ID for details",
                df_txns["transaction_id"].tolist() if "transaction_id" in df_txns.columns else []
            )

            if selected_id:
                detail = df_txns[df_txns["transaction_id"] == selected_id].iloc[0]
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.markdown("**Transaction Info**")
                    st.write(f"**ID:** {detail.get('transaction_id', 'N/A')}")
                    st.write(f"**Amount:** ₹{detail.get('transaction_amount', 0):,.2f}")
                    st.write(f"**Category:** {detail.get('merchant_category', 'N/A')}")
                    st.write(f"**Date:** {detail.get('transaction_datetime', 'N/A')}")
                    st.write(f"**Location:** {detail.get('transaction_location', 'N/A')}")
                with dc2:
                    st.markdown("**Customer Info**")
                    st.write(f"**Age:** {detail.get('customer_age', 'N/A')}")
                    st.write(f"**Income:** ₹{detail.get('customer_income_annual', 0):,}")
                    st.write(f"**Home:** {detail.get('customer_home_location', 'N/A')}")
                    st.write(f"**Risk Profile:** {detail.get('customer_risk_profile', 'N/A')}")
                    st.write(f"**Device:** {detail.get('device_type', 'N/A')}")
                with dc3:
                    st.markdown("**ML Results**")
                    badge = "🚨 Fraud" if detail.get("fraud_status") == "Fraud" else "✅ Genuine"
                    st.write(f"**Status:** {badge}")
                    st.write(f"**Fraud Probability:** {detail.get('fraud_probability', 0)*100:.1f}%")
                    st.write(f"**Risk Score:** {detail.get('fraud_risk_score', 0):.1f}/100")
                    st.write(f"**Suspicious:** {'⚡ Yes' if detail.get('is_suspicious') else 'No'}")
                    reasons = detail.get("top_reasons", "[]")
                    if isinstance(reasons, str):
                        try:
                            reasons = json.loads(reasons)
                        except Exception:
                            reasons = []
                    if reasons:
                        st.markdown("**Reasons:**")
                        for r in reasons:
                            st.write(f"• {r}")


# ─────────────────────────────────────────────────────────────
# PAGE 5: Model Insights
# ─────────────────────────────────────────────────────────────
elif page == "🧠 Model Insights":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🧠 Model Insights & Explainability</div>
        <div class="hero-subtitle">
            Comprehensive model performance metrics, feature importances, EDA visualizations, 
            and threshold configurations
        </div>
    </div>
    """, unsafe_allow_html=True)

    if engine.get("loaded"):
        metrics = engine["metrics_summary"]

        # Model Cards
        st.markdown('<div class="section-header">🏆 Best Models Selected</div>', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)

        with mc1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-value">Logistic Regression</div>
                <div class="kpi-label">Best Fraud Classifier</div>
                <div style="margin-top: 12px; color: #94a3b8; font-size: 0.85rem;">
                    Recall: 64.71% • ROC-AUC: 0.7148 • PR-AUC: 0.2029
                </div>
            </div>
            """, unsafe_allow_html=True)

        with mc2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📈</div>
                <div class="kpi-value">LightGBM Regressor</div>
                <div class="kpi-label">Best Risk Score Model</div>
                <div style="margin-top: 12px; color: #94a3b8; font-size: 0.85rem;">
                    R²: 0.8008 • MAE: 5.13 • RMSE: 6.90
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Classifier Comparison
        st.markdown('<div class="section-header">📊 Classification Model Comparison</div>', unsafe_allow_html=True)
        clf_data = metrics["classification_models"]
        clf_df = pd.DataFrame([
            {"Model": name, "Precision": m["precision"], "Recall": m["recall"],
             "F1": m["f1"], "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"]}
            for name, m in clf_data.items()
        ])

        fig = go.Figure()
        for metric_name in ["Recall", "ROC-AUC", "PR-AUC", "Precision"]:
            fig.add_trace(go.Bar(
                name=metric_name,
                x=clf_df["Model"],
                y=clf_df[metric_name],
                text=[f"{v:.3f}" for v in clf_df[metric_name]],
                textposition="outside",
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=420, barmode="group",
            title="Classification Metrics Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Regressor Comparison
        st.markdown('<div class="section-header">📈 Regression Model Comparison</div>', unsafe_allow_html=True)
        reg_data = metrics["regression_models"]
        reg_df = pd.DataFrame([
            {"Model": name, "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"]}
            for name, m in reg_data.items()
        ])

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            name="R²", x=reg_df["Model"], y=reg_df["R²"],
            marker_color="#6366f1", text=[f"{v:.4f}" for v in reg_df["R²"]],
            textposition="outside"
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            name="MAE", x=reg_df["Model"], y=reg_df["MAE"],
            mode="lines+markers", marker=dict(color="#f43f5e", size=10),
            line=dict(color="#f43f5e", width=2)
        ), secondary_y=True)
        fig.update_layout(**PLOTLY_LAYOUT, height=400, title="Regression Metrics (R² vs MAE)")
        fig.update_yaxes(title_text="R²", secondary_y=False, gridcolor="rgba(99,102,241,0.08)")
        fig.update_yaxes(title_text="MAE", secondary_y=True, gridcolor="rgba(99,102,241,0.08)")
        st.plotly_chart(fig, use_container_width=True)

        # Feature Importances
        st.markdown('<div class="section-header">🔬 Top Feature Importances (Classifier)</div>', unsafe_allow_html=True)
        fi = engine["feature_importances"]
        fi_df = pd.DataFrame([
            {"Feature": k, "Importance": v}
            for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)
        ])

        fig = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=["#6366f1", "#22d3ee", "#f43f5e"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=500, showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed", gridcolor="rgba(99,102,241,0.08)"))
        st.plotly_chart(fig, use_container_width=True)

        # Threshold Config
        st.markdown('<div class="section-header">⚙️ Threshold Configuration</div>', unsafe_allow_html=True)
        thresh = engine["threshold_config"]
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Fraud Probability Threshold</div>
                <div class="kpi-value">{thresh['fraud_probability_threshold']*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with tc2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Risk Score Suspicious Threshold</div>
                <div class="kpi-value">{thresh['risk_score_suspicious_threshold']:.0f}/100</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Suspicious Logic:** {thresh.get('suspicious_logic_description', 'N/A')}")

        # Training info
        st.markdown('<div class="section-header">📦 Training Summary</div>', unsafe_allow_html=True)
        ti1, ti2, ti3 = st.columns(3)
        with ti1:
            st.metric("Training Samples", f"{metrics['train_samples']:,}")
        with ti2:
            st.metric("Test Samples", f"{metrics['test_samples']:,}")
        with ti3:
            st.metric("Total Features", metrics["total_features"])

        # EDA Outputs
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Exploratory Data Analysis (EDA) Outputs</div>', unsafe_allow_html=True)

        eda_files = sorted(EDA_DIR.glob("*.png"))
        if eda_files:
            eda_names = {
                "eda_1": "Class Imbalance Analysis",
                "eda_2": "Transaction Amount vs Fraud",
                "eda_3": "Hourly Fraud Concentration",
                "eda_4": "Distance vs Fraud Density",
                "eda_5": "Location Mismatch vs Fraud",
                "eda_6": "Failed Attempts vs Fraud",
                "eda_7": "Correlation Heatmap",
                "eda_8": "Risk Score Distribution",
                "eda_9": "Feature Importances"
            }

            # Display in 3-column grid
            for i in range(0, len(eda_files), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(eda_files):
                        eda_file = eda_files[i + j]
                        prefix = eda_file.stem.rsplit("_", 1)[0] if "_" in eda_file.stem else eda_file.stem
                        # Get the eda_N prefix
                        parts = eda_file.stem.split("_")
                        key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else eda_file.stem
                        title = eda_names.get(key, eda_file.stem)
                        with col:
                            st.markdown(f"**{title}**")
                            st.image(str(eda_file), use_container_width=True)
        else:
            st.warning("No EDA outputs found. Run `train_pipeline.py` first.")
    else:
        st.error(f"ML Engine failed to load: {engine.get('error', 'Unknown error')}")


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div style="font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin-bottom: 6px; font-family: 'Outfit', sans-serif; letter-spacing: 0.05em;">
        🛡️ FRAUDSHIELD AI • SPATIAL RISK COGNITION SYSTEM
    </div>
    <div style="color: #94a3b8; margin-bottom: 8px; font-size: 0.8rem;">
        Dual ML Architecture: Calibrated Classifier + LightGBM Regressor + Real-Time Behavioral Heuristics
    </div>
    <div style="font-size: 0.7rem; color: #64748b; letter-spacing: 0.08em;">
        VISION-OS SPATIAL GLASS INTERFACE • TRANSACTION COGNITION MATRIX • STREAMLIT CORE
    </div>
</div>
""", unsafe_allow_html=True)
