import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import time
from model_utils import BrandModelEncoder

st.set_page_config(page_title="Watch Price Estimator", page_icon="⌚", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] { background-color: #080808; font-family: 'Inter', sans-serif; }
[data-testid="stHeader"] { background-color: #080808; }
[data-testid="stMain"] { padding-top: 0; }
[data-testid="stTabs"] { margin-top: 0; }
[data-testid="stTabContent"] { padding-top: 1.5rem; }

/* Tab styling */
[data-testid="stTabs"] button {
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #444 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.75rem 1.5rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #F5F0E8 !important;
    border-bottom-color: #F5F0E8 !important;
}

/* Labels */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    font-size: 0.62rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #555 !important;
    font-weight: 400 !important;
    margin-bottom: 4px !important;
}

/* Button */
[data-testid="stButton"] button {
    background-color: #F5F0E8 !important;
    color: #080808 !important;
    border: none !important;
    border-radius: 1px !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    width: 100% !important;
    margin-top: 1.25rem !important;
    padding: 0.85rem !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] button:hover { opacity: 0.85 !important; }

/* Hero */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    border-bottom: 1px solid #161616;
    margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    color: #444;
    text-transform: uppercase;
    margin-bottom: 1.25rem;
    font-family: 'Inter', sans-serif;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.6rem;
    font-weight: 300;
    color: #F5F0E8;
    line-height: 1.05;
    margin-bottom: 1rem;
    letter-spacing: -0.01em;
}
.hero-sub {
    font-size: 0.75rem;
    color: #555;
    letter-spacing: 0.08em;
}

/* Section labels */
.section-label {
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    color: #3A3A3A;
    text-transform: uppercase;
    margin-bottom: 1rem;
    margin-top: 2rem;
    padding-top: 1.5rem;
    font-family: 'Inter', sans-serif;
    position: relative;
}
.section-label::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(200,169,110,0.35) 50%, transparent 100%);
}

/* Result */
.result-wrap {
    margin-top: 2.5rem;
    padding-top: 2.5rem;
    text-align: center;
    position: relative;
}
.result-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(200,169,110,0.35) 50%, transparent 100%);
}
.result-eyebrow {
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    color: #444;
    text-transform: uppercase;
    margin-bottom: 1.25rem;
}
.result-price-wrap {
    position: relative;
    display: inline-block;
    padding: 1.2rem 1.8rem;
}
.result-price-wrap::before,
.result-price-wrap::after {
    content: '';
    position: absolute;
    width: 14px; height: 14px;
    border-color: rgba(200,169,110,0.5);
    border-style: solid;
}
.result-price-wrap::before { top: 0; left: 0; border-width: 1px 0 0 1px; }
.result-price-wrap::after  { top: 0; right: 0; border-width: 1px 1px 0 0; }
.corner-bl, .corner-br {
    position: absolute;
    width: 14px; height: 14px;
    border-color: rgba(200,169,110,0.5);
    border-style: solid;
    pointer-events: none;
}
.corner-bl { bottom: 0; left: 0; border-width: 0 0 1px 1px; }
.corner-br { bottom: 0; right: 0; border-width: 0 1px 1px 0; }
.result-price {
    font-family: 'Cormorant Garamond', serif;
    font-size: 5rem;
    font-weight: 300;
    color: #F5F0E8;
    letter-spacing: -0.02em;
    line-height: 1;
}
.result-price.shine {
    background: linear-gradient(90deg, #F5F0E8 0%, #F5F0E8 30%, #FFD700 50%, #F5F0E8 70%, #F5F0E8 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 1.5s ease forwards;
}
@keyframes shine {
    0% { background-position: 200% center; }
    100% { background-position: -200% center; }
}
.result-category {
    display: inline-block;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border: 1px solid #2A2A2A;
    color: #888;
    margin-top: 1rem;
    font-family: 'Inter', sans-serif;
}
.result-range {
    font-size: 0.72rem;
    color: #555;
    margin-top: 0.75rem;
    letter-spacing: 0.06em;
}

/* Specs grid */
.specs-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: #141414;
    border: 1px solid #141414;
    margin-top: 2rem;
    text-align: left;
}
.spec-item { background: #0C0C0C; padding: 1rem 1.25rem; }
.spec-label { font-size: 0.55rem; letter-spacing: 0.18em; color: #3A3A3A; text-transform: uppercase; margin-bottom: 0.4rem; }
.spec-value { font-size: 0.88rem; color: #888; font-weight: 300; font-family: 'Cormorant Garamond', serif; }

/* Warning */
.warning-box { background: #0C0C0C; border: 1px solid #1A1A1A; padding: 1rem; margin-top: 1rem; text-align: center; }
.warning-text { font-size: 0.72rem; color: #444; letter-spacing: 0.05em; }

/* Section note */
.section-note {
    font-size: 0.66rem;
    color: #555;
    letter-spacing: 0.04em;
    text-transform: none;
    margin-top: 0.3rem;
    margin-bottom: 1.2rem;
    font-family: 'Inter', sans-serif;
}

/* Insights */
.insight-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 300;
    color: #F5F0E8;
    margin-bottom: 0.4rem;
}
.insight-sub { font-size: 0.7rem; color: #444; letter-spacing: 0.06em; margin-bottom: 2rem; }
.insight-section { margin-top: 2.5rem; border-top: 1px solid #141414; padding-top: 1.5rem; }

/* Crystal glow */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 70vw; height: 70vh;
    background: radial-gradient(ellipse at center, rgba(200,169,110,0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Film grain */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    opacity: 0.04;
    pointer-events: none;
    z-index: 9999;
    animation: grain 6s steps(6) infinite;
}
@keyframes grain {
    0%   { transform: translate(0, 0); }
    17%  { transform: translate(-2%, -2%); }
    33%  { transform: translate(2%, 2%); }
    50%  { transform: translate(-1%, 1%); }
    67%  { transform: translate(1%, -1%); }
    83%  { transform: translate(-2%, 1%); }
    100% { transform: translate(0, 0); }
}
</style>
""", unsafe_allow_html=True)

# ── Background watch face ──────────────────────────────────────────────────────
st.markdown("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"
     style="position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
            width:65vmin;height:65vmin;opacity:0.06;pointer-events:none;z-index:0;">
  <circle cx="100" cy="100" r="97" fill="none" stroke="#F5F0E8" stroke-width="0.4"/>
  <circle cx="100" cy="100" r="90" fill="none" stroke="#F5F0E8" stroke-width="0.8"/>
  <circle cx="100" cy="100" r="85" fill="none" stroke="#F5F0E8" stroke-width="0.3"/>
  <g stroke="#F5F0E8" stroke-linecap="round">
    <line x1="100" y1="11" x2="100" y2="24" stroke-width="2.5"/>
    <line x1="100" y1="11" x2="100" y2="24" stroke-width="2.5" transform="rotate(90 100 100)"/>
    <line x1="100" y1="11" x2="100" y2="24" stroke-width="2.5" transform="rotate(180 100 100)"/>
    <line x1="100" y1="11" x2="100" y2="24" stroke-width="2.5" transform="rotate(270 100 100)"/>
    <line x1="100" y1="11" x2="100" y2="19" stroke-width="1.2" transform="rotate(30 100 100)"/>
    <line x1="100" y1="11" x2="100" y2="19" stroke-width="1.2" transform="rotate(60 100 100)"/>
    <line x1="100" y1="11" x2="100" y2="19" stroke-width="1.2" transform="rotate(120 100 100)"/>
    <line x1="100" y1="11" x2="100" y2="19" stroke-width="1.2" transform="rotate(150 100 100)"/>
    <line x1="100" y1="11" x2="100" y2="19" stroke-width="1.2" transform="rotate(210 100 100)"/>
    <line x1="100" y1="11" x2="100" y2="19" stroke-width="1.2" transform="rotate(240 100 100)"/>
    <line x1="100" y1="11" x2="100" y2="19" stroke-width="1.2" transform="rotate(300 100 100)"/>
    <line x1="100" y1="11" x2="100" y2="19" stroke-width="1.2" transform="rotate(330 100 100)"/>
  </g>
  <line x1="100" y1="100" x2="100" y2="44" stroke="#F5F0E8" stroke-width="1.8" stroke-linecap="round">
    <animateTransform attributeName="transform" type="rotate"
      from="0 100 100" to="360 100 100" dur="43200s" repeatCount="indefinite"/>
  </line>
  <line x1="100" y1="105" x2="100" y2="20" stroke="#F5F0E8" stroke-width="1" stroke-linecap="round">
    <animateTransform attributeName="transform" type="rotate"
      from="0 100 100" to="360 100 100" dur="3600s" repeatCount="indefinite"/>
  </line>
  <line x1="100" y1="115" x2="100" y2="15" stroke="#C8A96E" stroke-width="0.5" stroke-linecap="round">
    <animateTransform attributeName="transform" type="rotate"
      from="0 100 100" to="360 100 100" dur="60s" repeatCount="indefinite"/>
  </line>
  <circle cx="100" cy="100" r="3" fill="#F5F0E8"/>
  <circle cx="100" cy="100" r="1.5" fill="#C8A96E"/>
</svg>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
BRAND_MODELS = {
    "A. Lange & Söhne": ["1815","Cabaret","Datograph","Double Split","Grand Lange 1","Grand Langematik","Lange 1","Lange 31","Langematik","Langematik Perpetual","Little Lange 1","Odysseus","Richard Lange","Saxonia","Tourbograph","Zeitwerk"],
    "Audemars Piguet": ["Code 11.59","Edward Piguet","Jules Audemars","Millenary","Millenary 4101","Millenary Chronograph","Royal Oak","Royal Oak Chronograph","Royal Oak Concept","Royal Oak Day-Date","Royal Oak Dual Time","Royal Oak Jumbo","Royal Oak Lady","Royal Oak Offshore","Royal Oak Offshore Chronograph","Royal Oak Offshore Diver","Royal Oak Offshore Diver Chronograph","Royal Oak Offshore Lady","Royal Oak Perpetual Calendar","Royal Oak Selfwinding","Royal Oak Tourbillon"],
    "Breitling": ["Aerospace","Aerospace EVO","Avenger","Avenger II","Avenger II GMT","Aviator 8","Bentley GT","Chronomat","Chronomat 41","Chronomat 44","Chronomat 44 GMT","Chronomat Evolution","Chronoliner","Colt","Colt Automatic","Colt Chronograph","Emergency","Endurance Pro","Galactic","Navitimer","Navitimer 01","Navitimer 1 B01 Chronograph","Navitimer 8","Navitimer GMT","Navitimer Heritage","Navitimer World","Super Avenger","Superocean","Superocean Heritage","Superocean Heritage II 42","Transocean","Transocean Chronograph","Windrider"],
    "Cartier": ["Ballon Bleu","Ballon Bleu 33mm","Ballon Bleu 36mm","Ballon Bleu 40mm","Calibre de Cartier","Calibre de Cartier Chronograph","Calibre de Cartier Diver","Clé de Cartier","Drive de Cartier","Panthère","Pasha","Roadster","Ronde Solo de Cartier","Santos","Santos 100","Santos Dumont","Tank","Tank Américaine","Tank Anglaise","Tank Française","Tank Louis Cartier","Tank MC","Tank Solo","Tonneau","Trinity"],
    "Hamilton": ["Jazzmaster Viewmatic"],
    "Hublot": ["Big Bang","Big Bang 38 mm","Big Bang 41 mm","Big Bang 44 mm","Big Bang Ferrari","Big Bang King","Big Bang Meca-10","Big Bang Unico","Classic Fusion","Classic Fusion Aerofusion","Classic Fusion Chronograph","Classic Fusion Ultra-Thin","King Power","Spirit of Big Bang","Square Bang"],
    "IWC": ["Aquatimer","Aquatimer Automatic","Aquatimer Chronograph","Big Pilot","Big Pilot Top Gun","Da Vinci","Da Vinci Chronograph","Da Vinci Perpetual Calendar","Ingenieur","Ingenieur Automatic","Ingenieur Chronograph","Pilot","Pilot Chronograph","Pilot Mark","Pilot Worldtimer","Portofino","Portofino Automatic","Portofino Chronograph","Portuguese","Portuguese Annual Calendar","Portuguese Automatic","Portuguese Chronograph","Portuguese Perpetual Calendar","Portuguese Tourbillon","Portuguese Yacht Club Chronograph"],
    "Jaeger-LeCoultre": ["Deep Sea Chronograph","Duomètre","Geophysic True Second","Grande Reverso","Grande Reverso Calendar","Grande Reverso Ultra Thin","Master Calendar","Master Chronograph","Master Compressor","Master Compressor Chronograph","Master Compressor Diving","Master Control","Master Control Date","Master Eight Days Perpetual","Master Geographic","Master Grande Ultra Thin","Master Tourbillon","Master Ultra Thin","Master Ultra Thin Date","Master Ultra Thin Moon","Master Ultra Thin Perpetual","Odysseus","Polaris","Rendez-Vous","Reverso","Reverso Classique","Reverso Duetto","Reverso Duoface","Reverso Grande Taille","Reverso Lady","Reverso Squadra","Reverso Squadra Hometime"],
    "Longines": ["Admiral","Conquest","Conquest Classic","Conquest Heritage","DolceVita","Elegant","Evidenza","Flagship","Flagship Heritage","HydroConquest","La Grande Classique","Legend Diver","Master Collection","PrimaLuna","Présence","Record","Saint-Imier","Spirit"],
    "Montblanc": ["Sport"],
    "NOMOS": ["Club Neomatik","Tangente Neomatik"],
    "Omega": ["Constellation","Constellation Day-Date","Constellation Ladies","De Ville","De Ville Co-Axial","De Ville Hour Vision","De Ville Ladymatic","De Ville Prestige","De Ville Trésor","Globemaster","Seamaster","Seamaster 300","Seamaster Aqua Terra","Seamaster Diver 300 M","Seamaster Planet Ocean","Seamaster Planet Ocean Chronograph","Speedmaster","Speedmaster '57","Speedmaster Broad Arrow","Speedmaster Date","Speedmaster Professional Moonwatch","Speedmaster Racing","Speedmaster Reduced"],
    "Oris": ["Aquis","Aquis Chronograph","Aquis Date","Aquis GMT Date","Artelier","Artelier Chronograph","Artelier Date","Big Crown","Big Crown ProPilot","Big Crown ProPilot Chronograph","Big Crown ProPilot Date","Big Crown ProPilot GMT","Divers Sixty Five","ProDiver","ProDiver Chronograph"],
    "Panerai": ["Luminor","Luminor 1950","Luminor 1950 3 Days Chrono Flyback","Luminor 1950 3 Days GMT Automatic","Luminor Base","Luminor Due","Luminor GMT Automatic","Luminor Marina","Luminor Marina 1950 3 Days","Luminor Marina Automatic","Luminor Submersible","Mare Nostrum","Radiomir","Radiomir 1940","Radiomir 1940 3 Days","Radiomir 3 Days 47mm","Radiomir 8 Days","Radiomir Black Seal"],
    "Patek Philippe": ["Annual Calendar","Annual Calendar Chronograph","Aquanaut","Calatrava","Celestial","Chronograph","Complications","Golden Ellipse","Gondolo","Grand Complications","Grandmaster Chime","Nautilus","Perpetual Calendar","Perpetual Calendar Chronograph","Sky Moon Tourbillon","Travel Time","Twenty~4","Vintage","World Time"],
    "Rado": ["HyperChrome"],
    "Richard Mille": ["RM 005","RM 010","RM 011","RM 016","RM 027","RM 029","RM 030","RM 032","RM 035","RM 037","RM 052","RM 055","RM 07"],
    "Rolex": ["Air King","Cellini","Cellini Date","Cellini Dual Time","Datejust","Datejust 31","Datejust 36","Datejust 41","Datejust II","Day-Date","Day-Date 36","Day-Date 40","Day-Date II","Daytona","Explorer","Explorer II","GMT-Master","GMT-Master II","Lady-Datejust","Milgauss","Oyster Perpetual","Oyster Perpetual 28","Oyster Perpetual 31","Oyster Perpetual 36","Oyster Perpetual 41","Sea-Dweller","Sea-Dweller Deepsea","Sky-Dweller","Submariner","Submariner (No Date)","Submariner Date","Yacht-Master","Yacht-Master 40","Yacht-Master 42","Yacht-Master II"],
    "Seiko": ["5","5 Sports","Alpinist","Astron","Astron GPS Solar","Chronograph","Kinetic","Marinemaster","Monster","Premier","Presage","Prospex","Solar","Spirit"],
    "Sinn": ["103","104","140","142","144","203","240","256","303","356","358","456","556","656 / 657","756 / 757","856 / 857","900","903","956","EZM 3","U1","U1000","U2","UX"],
    "TAG Heuer": ["Aquaracer","Aquaracer 300M","Autavia","Carrera","Carrera Calibre 16","Carrera Calibre 1887","Carrera Calibre 5","Carrera Calibre 6","Carrera Calibre HEUER 01","Carrera Lady","Formula 1","Formula 1 Calibre 5","Formula 1 Lady","Grand Carrera","Link","Monaco","Monaco Calibre 11","Monaco Calibre 12","Monaco Calibre 6","Monaco V4","Monza","SLR"],
    "Tissot": ["Le Locle","PR 100","T-Touch","T-Race","Tradition","Visodate"],
    "Tudor": ["1926","Black Bay","Black Bay 36","Black Bay 41","Black Bay Bronze","Black Bay Chrono","Black Bay Fifty-Eight","Black Bay GMT","Black Bay S&G","Fastrider","Glamour","Glamour Date","Grantour","Grantour Chrono","Heritage Chrono","North Flag","Pelagos","Ranger","Submariner"],
    "Vacheron Constantin": ["Fiftysix","Historiques","Malte","Métiers d'Art","Overseas","Overseas Chronograph","Overseas Dual Time","Overseas World Time","Patrimony","Quai de l'Ile","Traditionnelle"],
    "Zenith": ["Captain","Captain Chronograph","Chronomaster Sport","Defy","Defy El Primero","El Primero","El Primero 36'000 VpH","El Primero Chronograph","El Primero Chronomaster","El Primero Sport","El Primero Stratos Flyback","Elite","Elite 6150","Elite Chronograph Classic","Elite Ultra Thin","Pilot","Pilot Type 20 Extra Special","Pilot Type 20 GMT","Port Royal","Star"],
}

CAT_MAPPINGS = {
    'movement':          ['Automatic','Manual winding','Quartz'],
    'case_material':     ['Aluminum','Bronze','Carbon','Ceramic','Gold/Steel','Palladium','Plastic','Platinum','Red gold','Rose gold','Silver','Steel','Tantalum','Titanium','Tungsten','White gold','Yellow gold'],
    'bracelet_material': ['Aluminium','Calf skin','Ceramic','Crocodile skin','Gold/Steel','Leather','Lizard skin','Ostrich skin','Plastic','Platinum','Red gold','Rose gold','Rubber','Satin','Shark skin','Silicon','Silver','Snake skin','Steel','Textile','Titanium','White gold','Yellow gold'],
    'condition':         ['Fair','Good','Incomplete','New','Poor','Unworn','Very good'],
    'sex':               ["Men's watch/Unisex","Women's watch"],
}

PLACEHOLDER = "— Please select —"

def price_category(price):
    if price < 5000:   return "Entry Luxury", "#4A4A4A"
    elif price < 20000: return "Mid Luxury", "#6A6A5A"
    elif price < 50000: return "High Luxury", "#8A7A5A"
    else:              return "Ultra Luxury", "#C8A96E"

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("watch_price_model.pkl")

pipeline = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">HSG · Skills: Introduction to Programming</div>
    <div class="hero-title">Watch Price<br>Estimator</div>
    <div class="hero-sub">Trained on 280,000+ listings · Chrono24</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Estimate", "Market Insights"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Brand and Model are the only required fields. All other fields are optional and default to the most common value in the dataset.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", [PLACEHOLDER] + sorted(BRAND_MODELS.keys()), index=0,
                             help="The watch manufacturer. Required.")
    with col2:
        model_opts = [PLACEHOLDER] + BRAND_MODELS.get(brand, []) if brand != PLACEHOLDER else [PLACEHOLDER]
        watch_model = st.selectbox("Model", model_opts, index=0,
                                   help="The specific reference within the brand. Required.")

    st.markdown('<div class="section-label">Materials</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Case and bracelet materials are strong price drivers. Leave blank to default to Steel.</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        case_material = st.selectbox("Case Material", [PLACEHOLDER] + CAT_MAPPINGS['case_material'], index=0,
                                     help="Material of the watch case. Defaults to Steel if left blank.")
    with col4:
        bracelet_material = st.selectbox("Bracelet Material", [PLACEHOLDER] + CAT_MAPPINGS['bracelet_material'], index=0,
                                         help="Material of the bracelet or strap. Defaults to Steel if left blank.")

    st.markdown('<div class="section-label">Specifications</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Movement type and condition significantly affect resale value. Size is set in whole millimetres using the slider below.</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        movement = st.selectbox("Movement", [PLACEHOLDER] + CAT_MAPPINGS['movement'], index=0,
                                help="The type of movement inside the watch. Defaults to Automatic.")
    with col6:
        condition = st.selectbox("Condition", [PLACEHOLDER] + CAT_MAPPINGS['condition'], index=0,
                                 help="The condition of the watch. Defaults to Very good.")

    col7, col8, col9 = st.columns(3)
    with col7:
        sex = st.selectbox("Category", [PLACEHOLDER] + CAT_MAPPINGS['sex'], index=0,
                           help="Intended wearer. Defaults to Men's / Unisex.")
    with col8:
        size = st.slider("Case Size (mm)", min_value=20, max_value=60, value=40, step=1,
                         help="Case diameter in whole millimetres.")
    with col9:
        year = st.number_input("Year", min_value=1950, max_value=2026, value=None, placeholder="e.g. 2018",
                               help="Year of production. Defaults to 2015 if left blank.")

    if st.button("Estimate Price"):
        missing = []
        if brand == PLACEHOLDER:       missing.append("Brand")
        if watch_model == PLACEHOLDER: missing.append("Model")

        if missing:
            st.markdown(f'<div class="warning-box"><div class="warning-text">Please select: {", ".join(missing)}</div></div>', unsafe_allow_html=True)
        else:
            use_movement          = movement if movement != PLACEHOLDER else 'Automatic'
            use_case_material     = case_material if case_material != PLACEHOLDER else 'Steel'
            use_bracelet_material = bracelet_material if bracelet_material != PLACEHOLDER else 'Steel'
            use_condition         = condition if condition != PLACEHOLDER else 'Very good'
            use_sex               = sex if sex != PLACEHOLDER else "Men's watch/Unisex"
            use_size              = float(size)
            use_year              = int(year) if year is not None else 2015

            input_data = pd.DataFrame([{
                'brand': brand, 'model': watch_model, 'movement': use_movement,
                'case_material': use_case_material, 'bracelet_material': use_bracelet_material,
                'yop': use_year, 'condition': use_condition, 'sex': use_sex, 'size': use_size
            }])

            for col in ['movement', 'case_material', 'bracelet_material', 'condition', 'sex']:
                input_data[col] = pd.Categorical(input_data[col], categories=CAT_MAPPINGS[col])

            log_price = pipeline.predict(input_data)[0]
            price = np.exp(log_price)
            low   = price * 0.85
            high  = price * 1.15
            cat_label, cat_color = price_category(price)

            # ── Counting animation ────────────────────────────────────────────
            result_placeholder = st.empty()
            steps = 40
            for i in range(steps + 1):
                progress = i / steps
                eased = progress ** 2
                current = price * eased
                result_placeholder.markdown(f"""
                <div class="result-wrap">
                    <div class="result-eyebrow">Estimated Market Value</div>
                    <div class="result-price">${current:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.05)

            # ── Final result with shine + category ────────────────────────────
            result_placeholder.markdown(f"""
            <div class="result-wrap">
                <div class="result-eyebrow">Estimated Market Value</div>
                <div class="result-price-wrap">
                    <span class="corner-bl"></span>
                    <span class="corner-br"></span>
                    <div class="result-price shine">${price:,.0f}</div>
                </div>
                <div class="result-category" style="border-color:{cat_color}; color:{cat_color};">{cat_label}</div>
                <div class="result-range">Estimated range &nbsp;·&nbsp; ${low:,.0f} — ${high:,.0f} USD</div>
                <div class="specs-grid">
                    <div class="spec-item"><div class="spec-label">Brand</div><div class="spec-value">{brand}</div></div>
                    <div class="spec-item"><div class="spec-label">Model</div><div class="spec-value">{watch_model}</div></div>
                    <div class="spec-item"><div class="spec-label">Condition</div><div class="spec-value">{use_condition}</div></div>
                    <div class="spec-item"><div class="spec-label">Year</div><div class="spec-value">{use_year}</div></div>
                    <div class="spec-item"><div class="spec-label">Case Material</div><div class="spec-value">{use_case_material}</div></div>
                    <div class="spec-item"><div class="spec-label">Case Size</div><div class="spec-value">{use_size} mm</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── SHAP explanation ──────────────────────────────────────────────
            st.markdown('<div class="section-label">Why this price?</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-note">Each bar shows how much a feature pushed the estimated price up (gold) or down (grey) relative to the dataset average.</div>', unsafe_allow_html=True)

            fitted_encoder = pipeline.named_steps["target_encoder"]
            fitted_xgb     = pipeline.named_steps["xgb"]

            input_for_shap = fitted_encoder.transform(input_data.copy())
            for col in ['movement', 'case_material', 'bracelet_material', 'condition', 'sex']:
                input_for_shap[col] = pd.Categorical(input_for_shap[col], categories=CAT_MAPPINGS[col])

            explainer   = shap.TreeExplainer(fitted_xgb)
            shap_values = explainer.shap_values(input_for_shap)

            feature_names = ['Brand', 'Model', 'Movement', 'Case Material',
                             'Bracelet Material', 'Year', 'Condition', 'Category', 'Size']
            shap_vals = shap_values[0]
            sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor('#0C0C0C')
            ax.set_facecolor('#0C0C0C')

            top_n = 6
            vals  = shap_vals[sorted_idx[:top_n]]
            names = [feature_names[i] for i in sorted_idx[:top_n]]
            colors = ['#C8A96E' if v > 0 else '#4A4A5A' for v in vals]

            bars = ax.barh(range(top_n), vals[::-1], color=colors[::-1], height=0.5)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(names[::-1], color='#888', fontsize=9, fontfamily='sans-serif')
            ax.set_xlabel('Impact on log(price)', color='#444', fontsize=8)
            ax.tick_params(colors='#444', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#1A1A1A')
            ax.spines['bottom'].set_color('#1A1A1A')
            ax.axvline(0, color='#222', linewidth=0.8)

            pos_patch = mpatches.Patch(color='#C8A96E', label='Increases price')
            neg_patch = mpatches.Patch(color='#4A4A5A', label='Decreases price')
            ax.legend(handles=[pos_patch, neg_patch], facecolor='#0C0C0C',
                      edgecolor='#1A1A1A', labelcolor='#666', fontsize=8)

            plt.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="insight-title">Market Insights</div>
    <div class="insight-sub">Based on 280,000+ Chrono24 listings</div>
    """, unsafe_allow_html=True)

    # ── Chart 1: Median price per brand ──────────────────────────────────────
    st.markdown('<div class="section-label">Median Price by Brand</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Median asking price in USD across all listings for each brand. Richard Mille and Patek Philippe anchor the ultra-luxury segment.</div>', unsafe_allow_html=True)

    brand_data = {
        'Richard Mille': 287000, 'Patek Philippe': 61750, 'Audemars Piguet': 48500,
        'A. Lange & Söhne': 37950, 'Vacheron Constantin': 19553, 'Rolex': 14670,
        'Hublot': 14499, 'Jaeger-LeCoultre': 8526, 'Panerai': 8018, 'IWC': 6979,
        'Zenith': 6717, 'Cartier': 5768, 'Breitling': 4914, 'Omega': 4900, 'Tudor': 3911
    }

    brands_sorted = sorted(brand_data.items(), key=lambda x: x[1])
    b_names = [b[0] for b in brands_sorted]
    b_vals  = [b[1] for b in brands_sorted]

    fig1, ax1 = plt.subplots(figsize=(7, 5.5))
    fig1.patch.set_facecolor('#0C0C0C')
    ax1.set_facecolor('#0C0C0C')

    norm = plt.Normalize(min(b_vals), max(b_vals))
    colors1 = plt.cm.YlOrBr(norm(b_vals))

    ax1.barh(b_names, b_vals, color=colors1, height=0.6)
    ax1.set_xlabel('Median Price (USD)', color='#444', fontsize=8)
    ax1.tick_params(colors='#666', labelsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#1A1A1A')
    ax1.spines['bottom'].set_color('#1A1A1A')

    for i, v in enumerate(b_vals):
        ax1.text(v + max(b_vals)*0.01, i, f'${v:,.0f}', va='center', color='#555', fontsize=7.5)

    ax1.set_yticks(range(len(b_names)))
    ax1.set_yticklabels(b_names, color='#888')
    plt.tight_layout()
    st.pyplot(fig1, width='stretch')
    plt.close()

    # ── Chart 2: Price by condition ───────────────────────────────────────────
    st.markdown('<div class="section-label">Median Price by Condition</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Unworn watches command the highest median price. Even a step from Unworn to Very Good represents a significant discount.</div>', unsafe_allow_html=True)

    cond_data = {
        'Unworn': 7746, 'Very good': 5578, 'New': 4900,
        'Good': 3021, 'Fair': 1747, 'Poor': 1214, 'Incomplete': 829
    }

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    fig2.patch.set_facecolor('#0C0C0C')
    ax2.set_facecolor('#0C0C0C')

    c_names = list(cond_data.keys())
    c_vals  = list(cond_data.values())
    bar_colors = ['#C8A96E' if i == 0 else '#3A3A3A' for i in range(len(c_names))]

    ax2.bar(c_names, c_vals, color=bar_colors, width=0.55)
    ax2.set_ylabel('Median Price (USD)', color='#444', fontsize=8)
    ax2.tick_params(colors='#666', labelsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#1A1A1A')
    ax2.spines['bottom'].set_color('#1A1A1A')

    plt.tight_layout()
    st.pyplot(fig2, width='stretch')
    plt.close()

    # ── Chart 3: Listings per brand ───────────────────────────────────────────
    st.markdown('<div class="section-label">Most Listed Brands</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Total number of listings in the training dataset. More listings means the model has seen more data for that brand and is likely more accurate.</div>', unsafe_allow_html=True)

    listings = {
        'Rolex': 72484, 'Omega': 39631, 'Seiko': 19297, 'Breitling': 18066,
        'Cartier': 16607, 'Longines': 15221, 'Audemars Piguet': 12987,
        'TAG Heuer': 12908, 'Hublot': 12750, 'Patek Philippe': 12397
    }

    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    fig3.patch.set_facecolor('#0C0C0C')
    ax3.set_facecolor('#0C0C0C')

    l_sorted = sorted(listings.items(), key=lambda x: x[1])
    ax3.barh([x[0] for x in l_sorted], [x[1] for x in l_sorted], color='#2A2A3A', height=0.6)
    ax3.set_xlabel('Number of Listings', color='#444', fontsize=8)
    ax3.tick_params(colors='#666', labelsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#1A1A1A')
    ax3.spines['bottom'].set_color('#1A1A1A')
    ax3.set_yticks(range(len(l_sorted)))
    ax3.set_yticklabels([x[0] for x in l_sorted], color='#888')

    plt.tight_layout()
    st.pyplot(fig3, width='stretch')
    plt.close()
