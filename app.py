import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Advisor | Mshauri wa Mkopo",
    page_icon="💡",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()
model   = bundle["model"]
enc     = bundle["encoder"]
feat_cols = bundle["feature_cols"]

# ── Translations ──────────────────────────────────────────────────────────────
T = {
    "en": {
        "title":           "Loan Recommendation Tool",
        "subtitle":        "Find out how much you could qualify for based on your business",
        "lang_label":      "Language / Lugha",
        "biz_type":        "What type of business do you run?",
        "biz_category":    "Business category",
        "biz_subcategory": "Business type",
        "region":          "Where is your business located?",
        "sales":           "What were your typical monthly sales?",
        "sales_help":      "Total money collected from customers in a typical month, before any expenses",
        "profits":         "What were your typical monthly profits?",
        "profits_help":    "Money left over after all expenses in a typical month",
        "startup":         "How much did you spend to start your business?",
        "startup_help":    "Total money spent to begin the business or buy it",
        "working_cap":     "What is the current value of your business stock, materials, and cash?",
        "working_help":    "Include inventory, supplies, and all cash/savings set aside for the business",
        "submit":          "Get my recommendation",
        "result_header":   "Your Loan Recommendation",
        "result_text":     "Based on businesses like yours, entrepreneurs typically qualify for:",
        "result_range":    "KES {low:,} – KES {high:,}",
        "result_typical":  "with a typical loan of **KES {mid:,}**",
        "result_note":     "This is a guide based on real loan data. Your actual offer depends on the bank's assessment.",
        "select":          "Select…",
    },
    "sw": {
        "title":           "Zana ya Ushauri wa Mkopo",
        "subtitle":        "Gundua kiasi unachoweza kustahili kulingana na biashara yako",
        "lang_label":      "Language / Lugha",
        "biz_type":        "Unafanya biashara ya aina gani?",
        "biz_category":    "Aina ya biashara",
        "biz_subcategory": "Aina ndogo ya biashara",
        "region":          "Biashara yako ipo wapi?",
        "sales":           "Mauzo yako ya kawaida kwa mwezi ni kiasi gani?",
        "sales_help":      "Jumla ya pesa zote ulizopokea kutoka kwa wateja kwa mwezi wa kawaida, kabla ya gharama",
        "profits":         "Faida yako ya kawaida kwa mwezi ni kiasi gani?",
        "profits_help":    "Pesa zilizobaki baada ya kutoa gharama zote kwa mwezi wa kawaida",
        "startup":         "Ulitumia kiasi gani kuanzisha biashara yako?",
        "startup_help":    "Jumla ya pesa uliyotumia kuanzisha au kununua biashara",
        "working_cap":     "Thamani ya sasa ya hisa, vifaa, na pesa za biashara yako ni nini?",
        "working_help":    "Jumuisha bidhaa, vifaa, na pesa/akiba zote za biashara",
        "submit":          "Pata ushauri wangu",
        "result_header":   "Ushauri Wako wa Mkopo",
        "result_text":     "Kulingana na biashara kama yako, wajasiriamali kwa kawaida wanastahili:",
        "result_range":    "KES {low:,} – KES {high:,}",
        "result_typical":  "mkopo wa kawaida ni **KES {mid:,}**",
        "result_note":     "Hii ni mwongozo kulingana na data halisi ya mikopo. Ofa yako halisi inategemea tathmini ya benki.",
        "select":          "Chagua…",
    },
}

# ── Taxonomy ──────────────────────────────────────────────────────────────────
TAXONOMY = {
    "Retail": {
        "en": "Retail",
        "sw": "Biashara ya Rejareja",
        "subs": {
            "grocery_food":      {"en": "Grocery & food staples",           "sw": "Duka la vyakula"},
            "clothing_fashion":  {"en": "Clothing & fashion",               "sw": "Nguo na mitindo"},
            "electronics":       {"en": "Electronics & phone accessories",  "sw": "Elektroniki na simu"},
            "pharmacy":          {"en": "Pharmacy / chemist",               "sw": "Duka la dawa"},
            "hardware":          {"en": "Hardware & building materials",    "sw": "Vifaa vya ujenzi"},
            "wines_spirits":     {"en": "Wines & spirits",                  "sw": "Pombe na vinywaji"},
            "cereals":           {"en": "Cereals & grains",                 "sw": "Nafaka na mazao"},
            "farm_inputs":       {"en": "Farm inputs / Agrovet",            "sw": "Dawa za kilimo"},
            "auto_spares":       {"en": "Auto & motorbike spares",          "sw": "Vipuri vya gari na pikipiki"},
            "beauty_products":   {"en": "Beauty products",                  "sw": "Bidhaa za urembo"},
            "books_stationery":  {"en": "Books & stationery",               "sw": "Vitabu na stesheni"},
            "retail_other":      {"en": "Other retail",                     "sw": "Rejareja nyingine"},
        },
    },
    "Services": {
        "en": "Services",
        "sw": "Huduma",
        "subs": {
            "salon_beauty":       {"en": "Salon & beauty services",         "sw": "Saluni na urembo"},
            "hotel_restaurant":   {"en": "Hotel / restaurant / food joint", "sw": "Hoteli na mgahawa"},
            "cyber_ict":          {"en": "Cyber café / IT services",        "sw": "Cyber na teknolojia"},
            "barbershop":         {"en": "Barbershop",                      "sw": "Kinyozi"},
            "rental_property":    {"en": "Rental property",                 "sw": "Kupangisha nyumba"},
            "taxi_transport":     {"en": "Taxi & transport",                "sw": "Teksi na usafiri"},
            "school_education":   {"en": "School / education",              "sw": "Shule na elimu"},
            "medical_health":     {"en": "Medical / health",                "sw": "Afya na matibabu"},
            "mpesa_agency":       {"en": "Mobile money / Mpesa agent",      "sw": "Mpesa na huduma za pesa"},
            "services_other":     {"en": "Other services",                  "sw": "Huduma nyingine"},
        },
    },
    "Agriculture": {
        "en": "Agriculture",
        "sw": "Kilimo",
        "subs": {
            "crop_farming":      {"en": "Crop farming",                     "sw": "Kulima mazao"},
            "livestock_poultry": {"en": "Livestock & poultry",              "sw": "Mifugo na kuku"},
            "dairy":             {"en": "Dairy",                            "sw": "Maziwa"},
            "fish":              {"en": "Fish",                             "sw": "Samaki"},
            "farm_inputs_agro":  {"en": "Farm inputs / Agrovet",            "sw": "Dawa na pembejeo za kilimo"},
            "agri_other":        {"en": "Other agriculture",                "sw": "Kilimo kingine"},
        },
    },
    "Wholesale": {
        "en": "Wholesale",
        "sw": "Biashara ya Jumla",
        "subs": {
            "wholesale_general":  {"en": "General wholesale",               "sw": "Jumla ya bidhaa"},
            "wholesale_cereals":  {"en": "Cereals wholesale",               "sw": "Jumla ya nafaka"},
            "wholesale_clothing": {"en": "Clothing wholesale",              "sw": "Jumla ya nguo"},
            "wholesale_hardware": {"en": "Hardware wholesale",              "sw": "Jumla ya vifaa"},
            "wholesale_other":    {"en": "Other wholesale",                 "sw": "Jumla nyingine"},
        },
    },
    "Manufacturing": {
        "en": "Manufacturing",
        "sw": "Utengenezaji",
        "subs": {
            "food_bakery":          {"en": "Food / bakery",                 "sw": "Chakula na mikate"},
            "tailoring_clothing":   {"en": "Tailoring & clothing",          "sw": "Ushonaji wa nguo"},
            "furniture_carpentry":  {"en": "Furniture & carpentry",         "sw": "Fanicha na useremala"},
            "metal_fabrication":    {"en": "Metal fabrication / welding",   "sw": "Chuma na useremala wa chuma"},
            "manuf_other":          {"en": "Other manufacturing",           "sw": "Utengenezaji mwingine"},
        },
    },
    "Transport": {
        "en": "Transport",
        "sw": "Usafirishaji",
        "subs": {
            "taxi_rideshare":   {"en": "Taxi / ride services",              "sw": "Teksi na usafiri"},
            "goods_transport":  {"en": "Goods transport",                   "sw": "Usafirishaji wa bidhaa"},
            "motorbike":        {"en": "Motorbike services",                "sw": "Huduma za pikipiki"},
            "petrol_station":   {"en": "Petrol station",                    "sw": "Kituo cha mafuta"},
            "transport_other":  {"en": "Other transport",                   "sw": "Usafirishaji mwingine"},
        },
    },
    "Construction": {
        "en": "Construction",
        "sw": "Ujenzi",
        "subs": {
            "hardware_materials":     {"en": "Hardware / building materials", "sw": "Vifaa vya ujenzi"},
            "carpentry":              {"en": "Carpentry",                     "sw": "Useremala"},
            "rental_construction":    {"en": "Rental property",              "sw": "Kupangisha nyumba"},
            "construction_services":  {"en": "Construction services",        "sw": "Huduma za ujenzi"},
            "construction_other":     {"en": "Other construction",           "sw": "Ujenzi mwingine"},
        },
    },
    "Banking_RealEstate": {
        "en": "Banking & Real Estate",
        "sw": "Benki na Mali Isiyohamishika",
        "subs": {
            "banking_agent":  {"en": "Banking agent",   "sw": "Wakala wa benki"},
            "real_estate":    {"en": "Real estate",     "sw": "Mali isiyohamishika"},
            "banking_other":  {"en": "Other",           "sw": "Nyingine"},
        },
    },
}

REGIONS = {
    "Rift-Valley":       {"en": "Rift Valley",       "sw": "Bonde la Ufa"},
    "Nairobi East":      {"en": "Nairobi East",      "sw": "Nairobi Mashariki"},
    "Nairobi West":      {"en": "Nairobi West",      "sw": "Nairobi Magharibi"},
    "North Eastern":     {"en": "North Eastern",     "sw": "Kaskazini Mashariki"},
    "Central":           {"en": "Central",           "sw": "Kati"},
    "Nyanza and Western":{"en": "Nyanza & Western",  "sw": "Nyanza na Magharibi"},
    "Coast":             {"en": "Coast",             "sw": "Pwani"},
}

# Financial range bands (label → midpoint)
SALES_BANDS = {
    "KES 0 – 100,000":          50_000,
    "KES 100,001 – 200,000":   150_000,
    "KES 200,001 – 300,000":   250_000,
    "KES 300,001 – 400,000":   350_000,
    "KES 400,001 – 500,000":   450_000,
    "KES 500,001 – 600,000":   550_000,
    "KES 600,001 – 700,000":   650_000,
    "KES 700,001 – 800,000":   750_000,
    "KES 800,001 – 900,000":   850_000,
    "KES 900,001 – 1,000,000": 950_000,
    "Above KES 1,000,000":   1_500_000,
}

PROFIT_BANDS = {
    "KES 0 – 25,000":         12_500,
    "KES 25,001 – 50,000":    37_500,
    "KES 50,001 – 75,000":    62_500,
    "KES 75,001 – 100,000":   87_500,
    "KES 100,001 – 125,000": 112_500,
    "KES 125,001 – 150,000": 137_500,
    "KES 150,001 – 175,000": 162_500,
    "KES 175,001 – 200,000": 187_500,
    "KES 200,001 – 225,000": 212_500,
    "KES 225,001 – 250,000": 237_500,
    "Above KES 250,000":     375_000,
}

CAPITAL_BANDS = {
    "Less than KES 10,000":         5_000,
    "KES 10,000 – 50,000":         30_000,
    "KES 50,001 – 100,000":        75_000,
    "KES 100,001 – 200,000":      150_000,
    "KES 200,001 – 500,000":      350_000,
    "More than KES 500,000":      750_000,
}

WC_BANDS = {
    "KES 0":                         0,
    "KES 1 – 50,000":           25_000,
    "KES 50,001 – 150,000":    100_000,
    "KES 150,001 – 400,000":   275_000,
    "KES 400,001 – 1,000,000": 700_000,
    "More than KES 1,000,000": 1_500_000,
}

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_loan(subcategory, region, sales, profits, startup_capital, working_capital):
    """Return (low, mid, high) loan amount in KES using percentiles across trees."""
    raw = pd.DataFrame([{
        "subcategory":     subcategory,
        "region":          region,
        "sales":           sales,
        "profits":         profits,
        "startup_capital": startup_capital,
        "working_capital": working_capital,
    }])
    raw[["subcategory", "region"]] = enc.transform(raw[["subcategory", "region"]])
    X_new = raw[feat_cols]

    # Collect predictions from each tree for an interval estimate
    tree_preds = np.array([tree.predict(X_new)[0] for tree in model.estimators_])
    low  = np.exp(np.percentile(tree_preds, 25))
    mid  = np.exp(np.percentile(tree_preds, 50))
    high = np.exp(np.percentile(tree_preds, 75))

    # Round to nearest 10,000 for clean display
    def r(x): return int(round(x / 10_000) * 10_000)
    return r(low), r(mid), r(high)

# ── UI ────────────────────────────────────────────────────────────────────────
# Language toggle
lang = st.sidebar.radio("🌐 Language / Lugha", ["English", "Kiswahili"])
L = "en" if lang == "English" else "sw"
t = T[L]

# Header
st.title(t["title"])
st.markdown(f"*{t['subtitle']}*")
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────

# 1. Business type (two-level dropdown)
st.subheader(t["biz_type"])
col1, col2 = st.columns(2)

with col1:
    cat_labels  = {k: v[L] for k, v in TAXONOMY.items()}
    cat_display = [t["select"]] + list(cat_labels.values())
    cat_choice  = st.selectbox(t["biz_category"], cat_display)

category_key = None
subcategory_key = None

with col2:
    if cat_choice == t["select"]:
        st.selectbox(t["biz_subcategory"], [t["select"]], disabled=True)
    else:
        category_key = [k for k, v in TAXONOMY.items() if v[L] == cat_choice][0]
        sub_labels   = {k: v[L] for k, v in TAXONOMY[category_key]["subs"].items()}
        sub_display  = [t["select"]] + list(sub_labels.values())
        sub_choice   = st.selectbox(t["biz_subcategory"], sub_display)
        if sub_choice != t["select"]:
            subcategory_key = [k for k, v in sub_labels.items() if v == sub_choice][0]

st.divider()

# 2. Region
region_labels  = {k: v[L] for k, v in REGIONS.items()}
region_display = [t["select"]] + list(region_labels.values())
region_choice  = st.selectbox(t["region"], region_display)
region_key     = None
if region_choice != t["select"]:
    region_key = [k for k, v in region_labels.items() if v == region_choice][0]

st.divider()

# 3–6. Financial inputs
st.subheader("📊 " + ("Your Business Finances" if L == "en" else "Fedha za Biashara Yako"))

sales_choice = st.selectbox(
    t["sales"],
    [t["select"]] + list(SALES_BANDS.keys()),
    help=t["sales_help"]
)

profits_choice = st.selectbox(
    t["profits"],
    [t["select"]] + list(PROFIT_BANDS.keys()),
    help=t["profits_help"]
)

startup_choice = st.selectbox(
    t["startup"],
    [t["select"]] + list(CAPITAL_BANDS.keys()),
    help=t["startup_help"]
)

wc_choice = st.selectbox(
    t["working_cap"],
    [t["select"]] + list(WC_BANDS.keys()),
    help=t["working_help"]
)

st.divider()

# ── Submit & result ───────────────────────────────────────────────────────────
all_filled = all([
    subcategory_key,
    region_key,
    sales_choice    != t["select"],
    profits_choice  != t["select"],
    startup_choice  != t["select"],
    wc_choice       != t["select"],
])

if st.button(t["submit"], type="primary", disabled=not all_filled):
    low, mid, high = predict_loan(
        subcategory    = subcategory_key,
        region         = region_key,
        sales          = SALES_BANDS[sales_choice],
        profits        = PROFIT_BANDS[profits_choice],
        startup_capital= CAPITAL_BANDS[startup_choice],
        working_capital= WC_BANDS[wc_choice],
    )

    st.success(f"### {t['result_header']}")
    st.markdown(t["result_text"])
    st.markdown(
        f"<h2 style='color:#1f6e3e; text-align:center'>"
        f"KES {low:,} – KES {high:,}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center'>{t['result_typical'].format(mid=mid)}</p>",
        unsafe_allow_html=True,
    )
    st.caption(t["result_note"])

elif not all_filled and st.session_state.get("submitted"):
    st.warning("Please fill in all fields." if L == "en" else "Tafadhali jaza sehemu zote.")
