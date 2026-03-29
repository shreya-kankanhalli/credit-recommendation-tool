import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thamini",
    page_icon="💚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 480px; }

  /* Step label */
  .step-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #F0B429;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
  }

  /* Big question text */
  .question {
    font-size: 1.45rem;
    font-weight: 700;
    color: #F0FFF4;
    line-height: 1.3;
    margin-bottom: 0.4rem;
  }

  /* Help text */
  .help-text {
    font-size: 0.88rem;
    color: #FDE68A;
    margin-bottom: 1.25rem;
    line-height: 1.5;
  }

  /* Primary button — bright green on dark bg */
  .stButton > button[kind="primary"] {
    background-color: #F0B429 !important;
    color: #0D3B1F !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    margin-top: 0.5rem;
  }
  .stButton > button[kind="primary"]:hover {
    background-color: #22C55E !important;
  }

  /* Secondary (back) button */
  .stButton > button[kind="secondary"] {
    background-color: transparent !important;
    color: #FDE68A !important;
    border: 2px solid #F0B429 !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.5rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
  }

  /* All other buttons (category/subcategory/region tap buttons) */
  .stButton > button:not([kind="primary"]):not([kind="secondary"]) {
    background-color: #143D22 !important;
    color: #F0FFF4 !important;
    border: 1.5px solid #2D6A4F !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.98rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    text-align: left !important;
    margin-bottom: 0.3rem;
  }
  .stButton > button:not([kind="primary"]):not([kind="secondary"]):hover {
    background-color: #1F5C35 !important;
    border-color: #F0B429 !important;
  }

  /* Progress bar */
  .stProgress > div > div { background-color: #F0B429 !important; }
  .stProgress > div { background-color: #143D22 !important; }

  /* Language toggle — prevent text wrap */
  [data-testid="stHorizontalBlock"] button {
    white-space: nowrap !important;
    overflow: hidden !important;
  }

  /* Selectbox */
  .stSelectbox > div > div {
    background-color: #143D22 !important;
    border-color: #2D6A4F !important;
    border-radius: 10px !important;
    color: #F0FFF4 !important;
  }

  /* Result */
  .result-range {
    font-size: 2.1rem;
    font-weight: 800;
    color: #F0B429;
    text-align: center;
    margin: 0.5rem 0;
  }
  .result-mid {
    font-size: 1rem;
    color: #FDE68A;
    text-align: center;
    margin-bottom: 1rem;
  }
  .result-note {
    font-size: 0.82rem;
    color: #FDE68A;
    text-align: center;
    line-height: 1.5;
    border-top: 1px solid #2D6A4F;
    padding-top: 1rem;
    margin-top: 1rem;
  }

  /* Welcome */
  .brand-title {
    font-size: 3rem;
    font-weight: 800;
    color: #F0B429;
    text-align: center;
    letter-spacing: -0.02em;
  }
  .brand-tagline {
    font-size: 1.1rem;
    color: #F0FFF4;
    text-align: center;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
    line-height: 1.5;
  }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

bundle    = load_model()
model     = bundle["model"]
enc       = bundle["encoder"]
feat_cols = bundle["feature_cols"]

# ── Translations ──────────────────────────────────────────────────────────────
T = {
    "en": {
        "tagline":      "Know what you qualify for",
        "welcome_body": "Answer 6 quick questions about your business and get a personalized loan estimate — based on data from Kenyan entrepreneurs.",
        "start":        "Get started",
        "back":         "← Back",
        "next":         "Next →",
        "step_of":      "Step {i} of {n}",
        "q_category":   "What type of business do you run?",
        "q_subcategory":"What best describes your business?",
        "q_region":     "Where is your business located?",
        "q_sales":      "What are your typical monthly sales?",
        "h_sales":      "Total money collected from customers in a typical month, before any expenses",
        "q_profits":    "What are your typical monthly profits?",
        "h_profits":    "Money left over after all expenses in a typical month",
        "q_startup":    "How much did you spend to start your business?",
        "h_startup":    "Total money invested to begin or buy the business",
        "q_working":    "What is the current value of your stock, materials, and business cash?",
        "h_working":    "Include all inventory, supplies, and cash/savings set aside for the business",
        "see_result":   "See my estimate",
        "result_title": "Your Loan Estimate",
        "result_intro": "Entrepreneurs with a business like yours typically qualify for:",
        "result_mid":   "with a typical amount of KES {mid:,}",
        "result_note":  "This estimate is based on real loan data from similar businesses. Your actual offer depends on the bank's assessment.",
        "restart":      "Start again",
        "select":       "Select…",
        "lang":         "🇬🇧 English",
    },
    "sw": {
        "tagline":      "Jua unachostahili",
        "welcome_body": "Jibu maswali 6 mafupi kuhusu biashara yako na upate makadirio ya mkopo — kulingana na data kutoka kwa wajasiriamali wa Kenya.",
        "start":        "Anza",
        "back":         "← Rudi",
        "next":         "Endelea →",
        "step_of":      "Hatua {i} kati ya {n}",
        "q_category":   "Unafanya biashara ya aina gani?",
        "q_subcategory":"Ni ipi inayoelezea biashara yako vizuri zaidi?",
        "q_region":     "Biashara yako ipo wapi?",
        "q_sales":      "Mauzo yako ya kawaida kwa mwezi ni kiasi gani?",
        "h_sales":      "Jumla ya pesa kutoka kwa wateja kwa mwezi wa kawaida, kabla ya gharama",
        "q_profits":    "Faida yako ya kawaida kwa mwezi ni kiasi gani?",
        "h_profits":    "Pesa zilizobaki baada ya kutoa gharama zote kwa mwezi wa kawaida",
        "q_startup":    "Ulitumia kiasi gani kuanzisha biashara yako?",
        "h_startup":    "Jumla ya pesa uliyotumia kuanzisha au kununua biashara",
        "q_working":    "Thamani ya sasa ya hisa, vifaa na pesa za biashara yako ni nini?",
        "h_working":    "Jumuisha bidhaa, vifaa, na pesa/akiba zote za biashara",
        "see_result":   "Ona makadirio yangu",
        "result_title": "Makadirio Yako ya Mkopo",
        "result_intro": "Wajasiriamali wenye biashara kama yako kwa kawaida wanastahili:",
        "result_mid":   "mkopo wa kawaida ni KES {mid:,}",
        "result_note":  "Makadirio haya yanategemea data halisi ya mikopo kutoka biashara zinazofanana. Ofa yako halisi inategemea tathmini ya benki.",
        "restart":      "Anza upya",
        "select":       "Chagua…",
        "lang":         "🇰🇪 Kiswahili",
    },
}

# ── Data ──────────────────────────────────────────────────────────────────────
TAXONOMY = {
    "Retail":             {"en": "Retail",                  "sw": "Biashara ya Rejareja",        "subs": {
        "grocery_food":       {"en": "Food & Grocery",           "sw": "Vyakula"},
        "clothing_fashion":   {"en": "Clothing & Fashion",       "sw": "Nguo"},
        "electronics":        {"en": "Electronics & Phones",     "sw": "Simu na vifaa vya Elektroniki"},
        "cereals":            {"en": "Cereals",                  "sw": "Nafaka na Mazao"},
        "farm_inputs":        {"en": "Farm Inputs",              "sw": "Vifaa vya kilimo"},
        "auto_spares":        {"en": "Auto & Motorbikes",        "sw": "Vipuri vya Gari"},
    }},
    "Services":           {"en": "Services",                "sw": "Huduma",                      "subs": {
        "salon_beauty":       {"en": "Salon & Beauty",           "sw": "Saluni na Urembo"},
        "hotel_restaurant":   {"en": "Hotel / Restaurant",       "sw": "Hoteli"},
        "cyber_ict":          {"en": "Cyber / IT",               "sw": "Cyber na I.T"},
        "barbershop":         {"en": "Barbershop",               "sw": "Kinyozi"},
        "rental_property":    {"en": "Rental Property",          "sw": "Nyumba za kukodisha"},
        "taxi_transport":     {"en": "Taxi & Transport",         "sw": "Teksi na uchukuzi"},
        "school_education":   {"en": "School / Education",       "sw": "Shule"},
        "medical_health":     {"en": "Health & Medical",         "sw": "Afya na Matibabu"},
        "mpesa_agency":       {"en": "Mpesa & Financial Services","sw": "Mpesa na Huduma za Pesa"},
        "services_other":     {"en": "Other Services",           "sw": "Huduma zinginezo"},
    }},
    "Agriculture":        {"en": "Agriculture",             "sw": "Kilimo",                      "subs": {
        "crop_farming":       {"en": "Crop Farming",             "sw": "Kilimo cha mimea"},
        "livestock_poultry":  {"en": "Livestock & Poultry",      "sw": "Mifugo na Kuku"},
        "dairy":              {"en": "Dairy",                    "sw": "Uzalishaji wa Maziwa"},
        "fish":               {"en": "Fish",                     "sw": "Samaki"},
        "farm_inputs_agro":   {"en": "Farm Inputs",              "sw": "Vifaa vya kilimo"},
        "agri_other":         {"en": "Other Agriculture",        "sw": "Aina zingine za kilimo"},
    }},
    "Wholesale":          {"en": "Wholesale",               "sw": "Biashara ya Jumla",           "subs": {
        "wholesale_general":  {"en": "General Wholesale",        "sw": "Bidhaa za Jumla"},
        "wholesale_cereals":  {"en": "Cereals",                  "sw": "Nafaka"},
        "wholesale_clothing": {"en": "Clothing",                 "sw": "Nguo"},
        "wholesale_hardware": {"en": "Hardware",                 "sw": "Vifaa vya Ujenzi"},
        "wholesale_other":    {"en": "Other",                    "sw": "Bidhaa zinginezo"},
    }},
    "Manufacturing":      {"en": "Manufacturing",           "sw": "Uundaji na Utengenezaji",     "subs": {
        "food_bakery":        {"en": "Food / Bakery",            "sw": "Vyakula / Mikate"},
        "tailoring_clothing": {"en": "Tailoring & Clothing",     "sw": "Ushonaji Nguo"},
        "furniture_carpentry":{"en": "Furniture & Carpentry",    "sw": "Fanicha na Useremala"},
        "metal_fabrication":  {"en": "Metal Fabrication",        "sw": "Uundaji kwa kutumia chuma"},
        "manuf_other":        {"en": "Other Manufacturing",      "sw": "Utengenezaji Mwingine"},
    }},
    "Transport":          {"en": "Transport",               "sw": "Uchukuzi na Ushafirishaji",   "subs": {
        "taxi_rideshare":     {"en": "Taxi",                     "sw": "Teksi"},
        "goods_transport":    {"en": "Goods Transport",          "sw": "Uchukuzi wa bidhaa"},
        "motorbike":          {"en": "Motorbike / Boda Boda",    "sw": "Pikipiki / Boda Boda"},
        "petrol_station":     {"en": "Petrol Station",           "sw": "Kituo cha mafuta"},
        "transport_other":    {"en": "Other Transport",          "sw": "Uchukuzi na ushafirishaji mwingine"},
    }},
    "Construction":       {"en": "Construction",            "sw": "Ujenzi",                      "subs": {
        "hardware_materials": {"en": "Hardware / Materials",     "sw": "Vifaa vya ujenzi"},
        "carpentry":          {"en": "Carpentry",                "sw": "Useremala"},
        "rental_construction":{"en": "Rental Property",          "sw": "Ujenzi wa Nyumba za kukodisha"},
        "construction_services":{"en":"Construction Services",   "sw": "Huduma za Ujenzi"},
        "construction_other": {"en": "Other Construction",       "sw": "Ujenzi mwingine"},
    }},
    "Banking_RealEstate": {"en": "Banking & Real Estate",   "sw": "Biashara ya Benki, Nyumba na Mashamba", "subs": {
        "banking_agent":      {"en": "Banking Agent",            "sw": "Ajenti wa Benki"},
        "real_estate":        {"en": "Real Estate",              "sw": "Nyumba na Mashamba"},
        "banking_other":      {"en": "Other",                    "sw": "Zinginezo"},
    }},
}

REGIONS = {
    "Nairobi":            {"en":"Nairobi",           "sw":"Nairobi"},
    "Central":            {"en":"Central",           "sw":"Kati"},
    "Rift-Valley":        {"en":"Rift Valley",       "sw":"Bonde la Ufa"},
    "Coast":              {"en":"Coast",             "sw":"Pwani"},
    "Nyanza":             {"en":"Nyanza",            "sw":"Nyanza"},
    "Western":            {"en":"Western",           "sw":"Magharibi"},
    "North Eastern":      {"en":"North Eastern",     "sw":"Kaskazini Mashariki"},
}

# Maps display region key → encoder key used during model training
REGION_ENCODE = {
    "Nairobi":      "Nairobi East",       # pooled; East used as proxy
    "Nyanza":       "Nyanza and Western", # displayed separately but same model key
    "Western":      "Nyanza and Western",
}

SALES_BANDS   = {"KES 0 – 100,000":50000,"KES 100,001 – 200,000":150000,"KES 200,001 – 300,000":250000,"KES 300,001 – 400,000":350000,"KES 400,001 – 500,000":450000,"KES 500,001 – 600,000":550000,"KES 600,001 – 700,000":650000,"KES 700,001 – 800,000":750000,"KES 800,001 – 900,000":850000,"KES 900,001 – 1,000,000":950000,"Above KES 1,000,000":1500000}
PROFIT_BANDS  = {"KES 0 – 25,000":12500,"KES 25,001 – 50,000":37500,"KES 50,001 – 75,000":62500,"KES 75,001 – 100,000":87500,"KES 100,001 – 125,000":112500,"KES 125,001 – 150,000":137500,"KES 150,001 – 175,000":162500,"KES 175,001 – 200,000":187500,"KES 200,001 – 225,000":212500,"KES 225,001 – 250,000":237500,"Above KES 250,000":375000}
CAPITAL_BANDS = {"Less than KES 10,000":5000,"KES 10,001 – 50,000":30000,"KES 50,001 – 100,000":75000,"KES 100,001 – 200,000":150000,"KES 200,001 – 500,000":350000,"More than KES 500,000":750000}
WC_BANDS      = {"KES 0":0,"KES 1 – 50,000":25000,"KES 50,001 – 150,000":100000,"KES 150,001 – 400,000":275000,"KES 400,001 – 1,000,000":700000,"More than KES 1,000,000":1500000}

TOTAL_STEPS = 6  # category, subcategory, region, sales, profits, startup, working capital
                 # (category+subcategory count as 2, so 7 data steps but displayed as 6 content steps)

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "step": 0,           # 0=welcome, 1-7=questions, 8=result
    "lang": "en",
    "category": None,
    "subcategory": None,
    "region": None,
    "sales": None,
    "profits": None,
    "startup": None,
    "working": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(step): st.session_state.step = step
def t(key):   return T[st.session_state.lang][key]

# ── Prediction ────────────────────────────────────────────────────────────────
def predict():
    region_key = REGION_ENCODE.get(st.session_state.region, st.session_state.region)
    raw = pd.DataFrame([{
        "subcategory":     st.session_state.subcategory,
        "region":          region_key,
        "sales":           SALES_BANDS[st.session_state.sales],
        "profits":         PROFIT_BANDS[st.session_state.profits],
        "startup_capital": CAPITAL_BANDS[st.session_state.startup],
        "working_capital": WC_BANDS[st.session_state.working],
    }])
    raw[["subcategory","region"]] = enc.transform(raw[["subcategory","region"]])
    X = raw[feat_cols]
    preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
    def r(x): return int(round(np.exp(x) / 10000) * 10000)
    return r(np.percentile(preds,25)), r(np.percentile(preds,50)), r(np.percentile(preds,75))

# ── Progress bar helper ───────────────────────────────────────────────────────
def progress_bar(step, total=7):
    pct = step / total
    st.progress(pct)
    st.markdown(
        f"<p class='step-label'>{t('step_of').format(i=step, n=total)}</p>",
        unsafe_allow_html=True
    )

# ── Language toggle (top right) ───────────────────────────────────────────────
_, lang_col = st.columns([2, 1])
with lang_col:
    other = "sw" if st.session_state.lang == "en" else "en"
    if st.button(T[other]["lang"], key="lang_toggle"):
        st.session_state.lang = other
        st.rerun()

step = st.session_state.step

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Welcome
# ══════════════════════════════════════════════════════════════════════════════
if step == 0:
    logo_path = "assets/ChatGPT Image Mar 29, 2026 at 07_58_06 PM.png"
    if os.path.exists(logo_path):
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.image(logo_path, use_container_width=True)
    else:
        st.markdown('<div style="font-size:5rem; text-align:center">🌱</div>',
                    unsafe_allow_html=True)

    st.markdown(f'<p class="brand-title">Thamini</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="brand-tagline">{t("tagline")}<br>'
        f'<span style="font-size:0.9rem; color:#888">{t("welcome_body")}</span></p>',
        unsafe_allow_html=True
    )
    if st.button(t("start"), type="primary"):
        go(1)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Business category
# ══════════════════════════════════════════════════════════════════════════════
elif step == 1:
    progress_bar(1)
    st.markdown(f'<p class="question">{t("q_category")}</p>', unsafe_allow_html=True)

    L = st.session_state.lang
    for key, val in TAXONOMY.items():
        if st.button(val[L], key=f"cat_{key}", use_container_width=True):
            st.session_state.category = key
            st.session_state.subcategory = None  # reset if changed
            go(2)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Business subcategory
# ══════════════════════════════════════════════════════════════════════════════
elif step == 2:
    progress_bar(2)
    L = st.session_state.lang
    cat = st.session_state.category
    cat_label = TAXONOMY[cat][L]
    st.markdown(
        f'<p class="step-label" style="color:#888">{cat_label}</p>'
        f'<p class="question">{t("q_subcategory")}</p>',
        unsafe_allow_html=True
    )

    for key, val in TAXONOMY[cat]["subs"].items():
        if st.button(val[L], key=f"sub_{key}", use_container_width=True):
            st.session_state.subcategory = key
            go(3)
            st.rerun()

    st.button(t("back"), type="secondary", on_click=go, args=(1,))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Region
# ══════════════════════════════════════════════════════════════════════════════
elif step == 3:
    progress_bar(3)
    L = st.session_state.lang
    st.markdown(f'<p class="question">{t("q_region")}</p>', unsafe_allow_html=True)

    for key, val in REGIONS.items():
        if st.button(val[L], key=f"reg_{key}", use_container_width=True):
            st.session_state.region = key
            go(4)
            st.rerun()

    st.button(t("back"), type="secondary", on_click=go, args=(2,))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Monthly sales
# ══════════════════════════════════════════════════════════════════════════════
elif step == 4:
    progress_bar(4)
    st.markdown(
        f'<p class="question">{t("q_sales")}</p>'
        f'<p class="help-text">{t("h_sales")}</p>',
        unsafe_allow_html=True
    )
    choice = st.selectbox(
        "", [t("select")] + list(SALES_BANDS.keys()),
        index=0 if not st.session_state.sales
               else list(SALES_BANDS.keys()).index(st.session_state.sales) + 1,
        label_visibility="collapsed"
    )
    col1, col2 = st.columns([1,2])
    with col1:
        st.button(t("back"), type="secondary", on_click=go, args=(3,))
    with col2:
        if st.button(t("next"), type="primary", disabled=(choice == t("select"))):
            st.session_state.sales = choice
            go(5)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Monthly profits
# ══════════════════════════════════════════════════════════════════════════════
elif step == 5:
    progress_bar(5)
    st.markdown(
        f'<p class="question">{t("q_profits")}</p>'
        f'<p class="help-text">{t("h_profits")}</p>',
        unsafe_allow_html=True
    )
    choice = st.selectbox(
        "", [t("select")] + list(PROFIT_BANDS.keys()),
        index=0 if not st.session_state.profits
               else list(PROFIT_BANDS.keys()).index(st.session_state.profits) + 1,
        label_visibility="collapsed"
    )
    col1, col2 = st.columns([1,2])
    with col1:
        st.button(t("back"), type="secondary", on_click=go, args=(4,))
    with col2:
        if st.button(t("next"), type="primary", disabled=(choice == t("select"))):
            st.session_state.profits = choice
            go(6)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Startup capital
# ══════════════════════════════════════════════════════════════════════════════
elif step == 6:
    progress_bar(6)
    st.markdown(
        f'<p class="question">{t("q_startup")}</p>'
        f'<p class="help-text">{t("h_startup")}</p>',
        unsafe_allow_html=True
    )
    choice = st.selectbox(
        "", [t("select")] + list(CAPITAL_BANDS.keys()),
        index=0 if not st.session_state.startup
               else list(CAPITAL_BANDS.keys()).index(st.session_state.startup) + 1,
        label_visibility="collapsed"
    )
    col1, col2 = st.columns([1,2])
    with col1:
        st.button(t("back"), type="secondary", on_click=go, args=(5,))
    with col2:
        if st.button(t("next"), type="primary", disabled=(choice == t("select"))):
            st.session_state.startup = choice
            go(7)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Working capital
# ══════════════════════════════════════════════════════════════════════════════
elif step == 7:
    progress_bar(7)
    st.markdown(
        f'<p class="question">{t("q_working")}</p>'
        f'<p class="help-text">{t("h_working")}</p>',
        unsafe_allow_html=True
    )
    choice = st.selectbox(
        "", [t("select")] + list(WC_BANDS.keys()),
        index=0 if not st.session_state.working
               else list(WC_BANDS.keys()).index(st.session_state.working) + 1,
        label_visibility="collapsed"
    )
    col1, col2 = st.columns([1,2])
    with col1:
        st.button(t("back"), type="secondary", on_click=go, args=(6,))
    with col2:
        if st.button(t("see_result"), type="primary", disabled=(choice == t("select"))):
            st.session_state.working = choice
            go(8)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Result
# ══════════════════════════════════════════════════════════════════════════════
elif step == 8:
    low, mid, high = predict()
    L = st.session_state.lang
    typical_label = "Typical loan amount" if L == "en" else "Mkopo wa kawaida"
    range_label   = "Likely range" if L == "en" else "Kiwango kinachowezekana"

    st.markdown(f'<p class="question" style="text-align:center">✅ {t("result_title")}</p>',
                unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; color:#FDE68A; margin-bottom:1.5rem">{t("result_intro")}</p>',
                unsafe_allow_html=True)

    # ── Typical amount (prominent, on top) ──
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:1.75rem">
      <div style="font-size:0.8rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
                  color:#FDE68A; margin-bottom:0.3rem">{typical_label}</div>
      <div style="font-size:3rem; font-weight:800; color:#F0B429; line-height:1.1">KES {mid:,}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Range bar (below) ──
    display_max = 2_000_000
    low_pct  = min(low  / display_max * 100, 96)
    mid_pct  = min(mid  / display_max * 100, 96)
    high_pct = min(high / display_max * 100, 96)
    fill_width = high_pct - low_pct

    st.markdown(f"""
    <div style="margin: 0 0 2rem 0">
      <div style="font-size:0.78rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
                  color:#FDE68A; margin-bottom:0.5rem">{range_label}</div>
      <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem">
        <span style="font-size:0.82rem; color:#FDE68A">KES {low:,}</span>
        <span style="font-size:0.82rem; color:#FDE68A">KES {high:,}</span>
      </div>
      <div style="position:relative; height:16px; background:#3D7A52; border-radius:99px;
                  box-shadow: inset 0 1px 3px rgba(0,0,0,0.4)">
        <div style="position:absolute; left:{low_pct:.1f}%; width:{fill_width:.1f}%;
                    height:100%; background:#F0B429; border-radius:99px;">
        </div>
        <div style="position:absolute; left:calc({mid_pct:.1f}% - 11px); top:-6px;
                    width:22px; height:28px; background:#F0B429; border-radius:7px;
                    box-shadow:0 0 14px rgba(240,180,41,1), 0 2px 6px rgba(0,0,0,0.4)">
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<p class="result-note">{t("result_note")}</p>', unsafe_allow_html=True)

    if st.button(t("restart"), type="secondary"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()
