import warnings, random, math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import base64
warnings.filterwarnings("ignore")


# Optional libs
HAVE_SM = False
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAVE_SM = True
except Exception:
    pass
HAVE_PULP = False
try:
    import pulp
    HAVE_PULP = True
except Exception:
    pass
# ---------- PAGE ----------
# ---------- Helper Function to Encode Image ----------
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        st.error(f"Image file '{image_path}' not found in the same directory as the script.")
        return None



st.set_page_config(page_title="In-Store Promotion Optimizer — Powered by AI", layout="wide")

image_path = "app_logo.png"
encoded_image = get_base64_image(image_path)




# ---------- PLOTLY THEME (full dark, all text white) ----------
dark_layout = go.Layout(
    paper_bgcolor="#0b0b0b",
    plot_bgcolor="#0b0b0b",
    font=dict(color="#ffffff"),
    title_font=dict(color="#ffffff"),
    legend=dict(bgcolor="#0b0b0b", font=dict(color="#ffffff")),
    xaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a", title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
    yaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a", title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
)
pio.templates["dark_custom"] = go.layout.Template(layout=dark_layout)
pio.templates.default = "dark_custom"
px.defaults.template = "dark_custom"
def apply_dark(fig: go.Figure, height: int = None, title: str = None):
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color="#ffffff")))
    if height:
        fig.update_layout(height=height)
    fig.update_layout(
        paper_bgcolor="#0b0b0b",
        plot_bgcolor="#0b0b0b",
        font_color="#ffffff",
        title_font_color="#ffffff",
        legend=dict(font=dict(color="#ffffff"), bgcolor="#0b0b0b"),
        xaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff"), gridcolor="#2a2a2a", zerolinecolor="#2a2a2a"),
        yaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff"), gridcolor="#2a2a2a", zerolinecolor="#2a2a2a"),
        annotations=[dict(font=dict(color="#ffffff")) for ann in fig.layout.annotations] if fig.layout.annotations else None,
        hoverlabel=dict(font_color="#ffffff", bgcolor="#1a1a1a")
    )
    # Apply text font color updates for specific trace types
    fig.update_traces(
        textfont=dict(color="#ffffff"),
        selector=dict(type='bar')
    )
    fig.update_traces(
        textfont=dict(color="#ffffff"),
        selector=dict(type='scatter')
    )
    fig.update_traces(
        textfont=dict(color="#ffffff"),
        selector=dict(type='pie')
    )
    fig.update_traces(
        textfont=dict(color="#ffffff"),
        insidetextfont=dict(color="#ffffff"),
        outsidetextfont=dict(color="#ffffff"),
        selector=dict(type='sunburst')
    )
    # Conditionally apply error_y only for scatter traces
    if any(trace.type == 'scatter' for trace in fig.data):
        fig.update_traces(
            error_y=dict(color="#ffffff"),
            selector=dict(type='scatter')
        )
    return fig
# ---------- CSS (dark, crisp, centered banner, smaller KPI cards) ----------
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
header {height: 0; visibility: hidden;}
html, body, .stApp { background: #0b0b0b; color: #efefef; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; }
.hero {
  background: radial-gradient(1000px 380px at 6% -20%, rgba(255,215,0,0.10), transparent 60%),
              linear-gradient(90deg, #0b0b0b, #121212 55%, #0b0b0b);
  border: 1px solid #2d2d2d; border-radius: 16px; padding: 20px 22px; margin-bottom: 12px;
  display: flex; align-items: center; justify-content: center;
}
.hero-content { text-align: center; max-width: 80%; }
.hero h1 { margin: 0; font-size: 32px; color: #FFD700; font-weight: 900; letter-spacing: .2px; }
.hero p { margin: 6px 0 0 0; color: #d8d8d8; font-size: 14.4px; }
.hero img { max-height: 80px; margin-left: 20px; }
.stTabs [role="tablist"]{ justify-content:center; gap: 10px; }
.stTabs [role="tab"]{
  font-size:15px; font-weight:900; color:#eaeaea; background:#151515;
  border:none; border-radius:12px; padding:10px 16px;
  box-shadow:0 2px 6px rgba(255,215,0,.12); transition:all .2s;
}
.stTabs [role="tab"]:hover{ color:#FFD700; transform: translateY(-1px); }
.stTabs [role="tab"][aria-selected="true"]{ color:#121212; background:linear-gradient(180deg,#FFD700,#f2c200); }
.stButton>button { background: linear-gradient(180deg, #252525, #1a1a1a); color: #fff; border:1px solid #3b3b3b; border-radius: 10px;
  padding: 10px 16px; font-weight: 800; letter-spacing:.2px; box-shadow: 0 2px 8px rgba(255,215,0,.16);}
.stButton>button:hover { border-color:#FFD700; color:#FFD700; }
.css-10trblm, .stSelectbox label, .stSlider label, .stMultiSelect label, .stNumberInput label, .stRadio label, .stTextInput label {
  color: #efefef !important; font-weight: 700 !important; letter-spacing: .2px;
}
.stSlider > div[data-baseweb="slider"] > div > div > div { color: #fff !important; }
.kpi { background: #141414; border: 1px solid #2d2d2d; border-radius: 12px; padding: 12px 14px; text-align:center; }
.kpi h5 { margin: 0; color: #ffd86a; font-size: 11px; font-weight: 900; letter-spacing: .5px; text-transform: uppercase; }
.kpi h2 { margin: 6px 0 0; color: #fff; font-size: 22px; font-weight: 900; }
.kpi small { color:#bdbdbd; font-size: 10px; }
.card, .deck-card { background: #131313; border: 1px solid #2a2a2a; border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; }
.deck-card h3 { margin:2px 0 8px 0; color:#FFD700; font-size:18px; font-weight:900; }
.deck-card h4 { margin:0 0 6px; font-weight:900; color:#eaeaea; }
.helpbox{ border:1px solid #3a3a3a; border-radius:12px; padding:10px 14px; background:#141414; color:#ddd; font-size:13px; }
.insight{ border:1px dashed #FFD700; border-radius:12px; padding:10px 14px; background:#171510; color:#ffd86a; font-size:13px; }
.section-title{ font-weight:900; font-size:18px; color:#FFD700; margin-top:6px; }
.flowbox{ background:#131313; border:1px solid #2d2d2d; border-radius:12px; padding:14px }
.flow-step{ display:inline-block; background:#191919; border:1px solid #333; padding:10px 12px; border-radius:10px; margin:6px; font-weight:800 }
.arrow{ display:inline-block; padding:0 8px; color:#ffd86a; font-weight:900 }
</style>
""", unsafe_allow_html=True)
# ---------- HERO (centered with image from repository) ----------
if encoded_image:
    st.markdown(f"""
    <div class="hero">
      <div class="hero-content">
        <h1>In-Store Promotion Optimizer — Powered by AI</h1>
        <p>Answer-first analytics to <b>choose the right SKUs</b>, <b>set the right discounts</b>, and <b>monetize uplift</b> across regions and categories.</p>
      </div>
      <img src="{encoded_image}" alt="Promo Image">
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero">
      <div class="hero-content">
        <h1>In-Store Promotion Optimizer — Powered by AI </h1>
        <p>Answer-first analytics to <b>choose the right SKUs</b>, <b>set the right discounts</b>, and <b>monetize uplift</b> across regions and categories.</p>
      </div> 
      <img src="https://via.placeholder.com/80x80?text=Promo" alt="Promo Image">
    </div>
    """, unsafe_allow_html=True)
# CONSTANTS & CATALOG
# =========================
REGIONS_BASE = ["Dubai", "Abu Dhabi", "Northern Emirates"]
REGION_OPTIONS = ["All Regions"] + REGIONS_BASE
MECHS = {"PCT10":0.10, "PCT15":0.15, "PCT20":0.20, "BOGO":0.33}
CATS = {
    "Rice": ["Basmati","Jasmine","Calrose","Brown","Black","Parboiled","Short Grain","Organic"],
    "Oils": ["Sunflower","Olive","Canola","Corn","Blended","Coconut","Ghee","Avocado"],
    "Pasta": ["Spaghetti","Penne","Fusilli","Macaroni","Lasagna","Vermicelli","Farfalle","Rigatoni"],
    "Snacks": ["Chips","Nuts","Popcorn","Biscuits","Granola Bars","Crackers","Nachos","Cookies"],
    "Beverages": ["Cola","Juice","Water","Energy","Iced Tea","Sparkling","Isotonic","Flavored Water"],
    "Dairy": ["Milk","Yogurt","Cheese","Butter","Cream","Paneer","Laban","Ghee"],
    "Cleaning": ["Detergent","Dishwash","Floor","Bleach","Toilet","Glass","Multipurpose","Fabric Softener"],
    "Produce": ["Tomato","Onion","Banana","Apple","Cucumber","Potato","Orange","Grapes"],
    "Bakery": ["Bread","Buns","Croissants","Cakes","Muffins","Tortilla","Pita","Rusk"],
    "Breakfast": ["Cereal","Oats","Muesli","Pancake Mix","Honey","Jam","Peanut Butter","Chocolate Spread"]
}
PRICE_TIERS = ["Value","Mainstream","Premium"]
def _seed(s=42):
    np.random.seed(s); random.seed(s)
@st.cache_data(show_spinner=False)
def generate_catalog(seed:int=42, skus_per_cat:int=10) -> pd.DataFrame:
    _seed(seed)
    brands=["BrandA","BrandB","BrandC","BrandD","BrandE","PrivateLabel","BrandF","BrandG"]
    rows=[]
    for cat, subcats in CATS.items():
        # Ensure exactly skus_per_cat SKUs per category
        for i in range(skus_per_cat):
            sub = np.random.choice(subcats)
            tier = np.random.choice(PRICE_TIERS, p=[0.25,0.55,0.20])
            base_price = {"Value":np.random.uniform(2,7),
                          "Mainstream":np.random.uniform(6,22),
                          "Premium":np.random.uniform(15,45)}[tier]
            cogs = base_price*np.random.uniform(0.6,0.8)
            brand = np.random.choice(brands)
            elasticity = {
                "Rice":-1.2,"Oils":-1.0,"Pasta":-1.1,"Snacks":-1.6,"Beverages":-1.5,
                "Dairy":-0.9,"Cleaning":-1.2,"Produce":-0.6,"Bakery":-0.8,"Breakfast":-1.1
            }[cat] + np.random.normal(0,0.25)
            rows.append({
                "sku_id": f"{cat[:2].upper()}-{i+1:03d}",
                "category": cat, "subcategory": sub, "brand": brand, "price_tier": tier,
                "base_price": round(base_price,2), "cogs": round(cogs,2),
                "elasticity_true": round(float(elasticity),2)
            })
    return pd.DataFrame(rows)
@st.cache_data(show_spinner=False)
def generate_stores(seed:int=42, n_stores:int=50) -> pd.DataFrame:
    _seed(seed)
    stores=[]
    for i in range(n_stores):
        r = np.random.choice(REGIONS_BASE, p=[0.55,0.25,0.20])
        fmt = np.random.choice(["Hyper","Super"], p=[0.35,0.65])
        stores.append({"store_id":f"S{str(i+1).zfill(3)}","region":r,"format":fmt})
    return pd.DataFrame(stores)
@st.cache_data(show_spinner=True, ttl=3600)
def generate_epos(seed:int=42, weeks:int=26, promo_rate=0.06):
    _seed(seed)
    catalog = generate_catalog(seed)
    stores = generate_stores(seed)
    start = datetime(2024,1,1)
    week_ids=[f"W{w+1:03d}" for w in range(weeks)]
    week_dates=[(start+timedelta(weeks=w)).date() for w in range(weeks)]
    ramadan_weeks = list(range(12,17)) if weeks >= 17 else []
    summer_weeks = list(range(26,36)) if weeks >= 36 else []
    rows=[]
    for _, s in stores.iterrows():
        rfac = {"Dubai":1.25,"Abu Dhabi":1.05,"Northern Emirates":0.85}[s.region]
        ffac = {"Hyper":1.35,"Super":1.0}[s.format]
        scale = rfac*ffac
        for _, k in catalog.iterrows():
            base_units = np.clip(np.random.normal(6,3),1,30)
            base_units *= (1.1 if k["category"] in ["Rice","Dairy","Breakfast"] else 1.0)
            base_units *= (1.0 if k["price_tier"]=="Mainstream" else (0.8 if k["price_tier"]=="Premium" else 0.9))
            for w in range(weeks):
                season_mult = 1.0
                if (w+1) in ramadan_weeks and k["category"] in ["Rice","Oils","Breakfast","Dairy"]:
                    season_mult *= 1.12
                if (w+1) in summer_weeks and k["category"] in ["Beverages","Snacks"]:
                    season_mult *= 1.18
                base_price = k["base_price"]*np.random.uniform(0.985,1.015)
                promo_flag=0; promo_mech="NONE"; disc=0.0
                if np.random.rand() < (promo_rate + (0.05 if k["category"] in ["Rice","Snacks","Beverages"] else 0.0)):
                    promo_flag=1
                    promo_mech=np.random.choice(list(MECHS.keys()), p=[.35,.35,.25,.05])
                    disc=MECHS[promo_mech]
                price = base_price*(1.0-disc)
                beta = k["elasticity_true"]
                units = base_units*scale*season_mult*(price/k["base_price"])**beta
                units = np.random.poisson(max(1, units))
                pull_next=0
                if promo_flag and k["category"] in ["Rice","Oils","Breakfast"] and disc>=0.15:
                    pull_next = int(units*np.random.uniform(0.05,0.12))
                rows.append({
                    "week_id": week_ids[w], "date": week_dates[w],
                    "store_id": s["store_id"], "region": s["region"], "format": s["format"],
                    "sku_id": k["sku_id"], "category": k["category"], "subcategory": k["subcategory"],
                    "price": round(float(price),2), "promo_flag": promo_flag, "promo_mechanic": promo_mech,
                    "qty": int(units), "pull_forward_next": pull_next
                })
    epos = pd.DataFrame(rows)
    epos = epos.merge(catalog[["sku_id","brand","price_tier","base_price","cogs"]], on="sku_id", how="left")
    epos["revenue"] = epos["price"]*epos["qty"]
    epos["margin"] = (epos["price"]-epos["cogs"]).clip(lower=0)*epos["qty"]
    epos["gm%"] = np.where(epos["revenue"]>0, epos["margin"]/epos["revenue"], 0.0)
    return catalog, stores, epos, week_ids
# =========================
# FEATURE HELPERS
# =========================
@st.cache_data(show_spinner=False)
def compute_cv_adi(epos:pd.DataFrame):
    g = epos.groupby(["region","store_id","sku_id"]).agg(
        mean_qty=("qty","mean"), std_qty=("qty","std"),
        weeks=("qty","count"), nonzero=("qty", lambda x:(x>0).sum())
    ).reset_index()
    g["cv"] = (g["std_qty"]/g["mean_qty"].replace(0,np.nan)).fillna(0).clip(0,5)
    vals=[]
    for (reg, store, sku), df in epos.sort_values("date").groupby(["region","store_id","sku_id"]):
        idx = df.index[df["qty"]>0]
        if len(idx)<2: adi = df.shape[0]
        else: adi = float(np.diff(idx).mean())
        vals.append({"region":reg,"store_id":store,"sku_id":sku,"adi":adi})
    return g.merge(pd.DataFrame(vals), on=["region","store_id","sku_id"], how="left")
def quick_elasticity(epos, sku_id, region=None):
    df = epos[epos["sku_id"]==sku_id].copy()
    if region and region != "All Regions":
        df = df[df["region"]==region].copy()
    df = df[(df["qty"]>0) & (df["price"]>0)]
    if len(df)<12: return None
    df["logq"]=np.log(df["qty"]); df["logp"]=np.log(df["price"])
    b = np.polyfit(df["logp"].values, df["logq"].values, 1)[0]
    return float(b)
def net_impact_components(epos, sku_id, week_id, region):
    sub = epos.copy()
    if region != "All Regions":
        sub = sub[sub["region"]==region]
    dfw = sub[(sub["week_id"]==week_id)]
    dfi = dfw[dfw["sku_id"]==sku_id]
    if dfi.empty: return dict(lift=0, halo=0, cannib=0, pull=0)
    cat = dfi["category"].iloc[0]
    hist = sub[(sub["sku_id"]==sku_id)&(sub["promo_flag"]==0)]
    base_margin = hist["margin"].median() if len(hist)>0 else 0
    lift = dfi["margin"].sum() - base_margin
    halo_pairs = {("Snacks","Beverages"):0.05, ("Rice","Oils"):0.05, ("Bakery","Breakfast"):0.04}
    cannib_pct = {"Rice":0.12,"Oils":0.1,"Pasta":0.11,"Snacks":0.16,"Beverages":0.15,"Dairy":0.09,
                  "Cleaning":0.12,"Produce":0.06,"Bakery":0.10,"Breakfast":0.11}.get(cat,0.1)
    halo=0
    for (a,b),eff in halo_pairs.items():
        if cat==a:
            halo += dfw[dfw["category"]==b]["margin"].sum()*eff*0.1
    cannib = dfw[dfw["category"]==cat]["margin"].sum()*cannib_pct*0.1
    nextweek_num = int(str(week_id)[1:])
    wk_all = pd.to_numeric(epos['week_id'].astype(str).str.extract(r"W(\d+)")[0], errors="coerce")
    max_wk = int(wk_all.dropna().max()) if wk_all.notna().any() else nextweek_num
    next_w = f"W{min(nextweek_num+1, max_wk):03d}"
    pf_rows = sub[(sub["sku_id"]==sku_id)&(sub["week_id"]==next_w)]
    if len(pf_rows)>0 and dfi["qty"].sum()>0:
        pull_units = pf_rows["pull_forward_next"].sum()
        avg_mpu = (dfi["margin"].sum()/max(1,dfi["qty"].sum()))
        pull = pull_units*avg_mpu
    else: pull=0
    return dict(lift=float(lift), halo=float(halo), cannib=float(cannib), pull=float(pull))
def _ensure_region(df: pd.DataFrame, region_sel):
    if region_sel == "All Regions":
        return df.copy()
    return df[df["region"]==region_sel].copy()
def tlearner_uplift(epos:pd.DataFrame, catalog:pd.DataFrame, region=None, category=None, recent_weeks:int=8, bootstrap:int=20):
    try:
        df = epos.copy()
        if region and region != "All Regions":
            df = df[df["region"]==region]
        if category:
            df = df[df["category"]==category]
        if len(df)<500:
            return pd.DataFrame()
        wk = df["week_id"].astype(str).str.extract(r"W(\\d+)")[0]
        wk = pd.to_numeric(wk, errors="coerce")
        if wk.isna().all():
            df = df.sort_values("date").copy()
            df["week_num"] = np.arange(1, len(df)+1)
        else:
            df["week_num"] = wk.fillna(method="ffill").fillna(method="bfill").astype(int)
        df["month"] = pd.to_datetime(df["date"]).dt.month
        feats_cat = ["brand","price_tier","category"]
        feats_num = ["price","week_num","month"]
        y = df["margin"].values
        T = df["promo_flag"]==1
        C = ~T
        if T.sum()<150 or C.sum()<150:
            proxy = []
            for sid, d in df.groupby("sku_id"):
                base = d[d["promo_flag"]==0]
                if len(base)<5: continue
                beta = quick_elasticity(df, sid)
                if beta is None: beta = -1.2
                bp = d["base_price"].iloc[0]; cogs = d["cogs"].iloc[0]
                base_units = max(1, base["qty"].tail(8).mean())
                price_15 = bp*(1-0.15)
                units_15 = base_units*(price_15/bp)**beta
                marg_base = (bp-cogs)*base_units
                marg_15 = (price_15-cogs)*units_15
                proxy.append([sid, marg_15 - marg_base, (base["margin"].tail(8).mean() if len(base)>0 else 0.0)])
            if not proxy:
                return pd.DataFrame()
            out = pd.DataFrame(proxy, columns=["sku_id","pred_incremental_margin","baseline_margin"])
            out = out.merge(catalog[["sku_id","brand","price_tier","category","base_price","cogs"]], on="sku_id", how="left")
            out["upl_ci_low"] = out["pred_incremental_margin"]*0.8
            out["upl_ci_high"]= out["pred_incremental_margin"]*1.2
            out["method"] = "elasticity-proxy"
            return out
        pre = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), feats_cat),
             ("num", "passthrough", feats_num)]
        )
        model_T = Pipeline([("pre", pre), ("rf", RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2))])
        model_C = Pipeline([("pre", pre), ("rf", RandomForestRegressor(n_estimators=100, random_state=43, min_samples_leaf=2))])
        model_T.fit(df.loc[T, feats_cat+feats_num], y[T])
        model_C.fit(df.loc[C, feats_cat+feats_num], y[C])
        latest_w = df["week_num"].max()
        X0 = df[df["week_num"]>=latest_w-(recent_weeks-1)][feats_cat+feats_num]
        sk = df[df["week_num"]>=latest_w-(recent_weeks-1)]["sku_id"].values
        if len(X0)==0:
            return pd.DataFrame()
        preds_T = model_T.predict(X0); preds_C = model_C.predict(X0)
        upl = pd.DataFrame({"sku_id": sk, "upl": preds_T - preds_C})
        base = df[(df["week_num"]>=latest_w-(recent_weeks-1))&(df["promo_flag"]==0)].groupby("sku_id").agg(baseline_margin=("margin","mean")).reset_index()
        out = upl.groupby("sku_id").agg(pred_incremental_margin=("upl","mean")).reset_index()
        out = out.merge(base, on="sku_id", how="left").fillna({"baseline_margin":0.0})
        out = out.merge(catalog[["sku_id","brand","price_tier","category","base_price","cogs"]], on="sku_id", how="left")
        rng = np.random.default_rng(42)
        ci_low=[]; ci_high=[]
        for sid in out["sku_id"]:
            sub = upl[upl["sku_id"]==sid]["upl"].values
            if len(sub)<5:
                ci_low.append(out.loc[out["sku_id"]==sid, "pred_incremental_margin"].values[0]*0.8)
                ci_high.append(out.loc[out["sku_id"]==sid, "pred_incremental_margin"].values[0]*1.2)
                continue
            boots=[]
            for _ in range(bootstrap):
                samp = rng.choice(sub, size=len(sub), replace=True)
                boots.append(samp.mean())
            ci_low.append(np.percentile(boots, 10))
            ci_high.append(np.percentile(boots, 90))
        out["upl_ci_low"] = ci_low
        out["upl_ci_high"]= ci_high
        out["method"] = "t-learner"
        return out
    except Exception as e:
        st.error(f"Uplift modeling error: {e}")
        return pd.DataFrame()
def ilp_leaflet(uplift_df, slots=24, per_cat_min=1, per_brand_cap=3, budget=None, price_map=None):
    if (uplift_df is None) or uplift_df.empty:
        return pd.DataFrame(), 0.0, "No uplift data."
    if not HAVE_PULP:
        return pd.DataFrame(), 0.0, "PuLP not installed."
    df = uplift_df.copy().drop_duplicates("sku_id")
    df["pred_incremental_margin"] = df["pred_incremental_margin"].clip(lower=0)
    prob = pulp.LpProblem("Leaflet", pulp.LpMaximize)
    x = {row.sku_id: pulp.LpVariable(f"x_{row.sku_id}", lowBound=0, upBound=1, cat="Binary")
         for _, row in df.iterrows()}
    prob += pulp.lpSum([x[r.sku_id]*r.pred_incremental_margin for _, r in df.iterrows()])
    prob += pulp.lpSum([x[s] for s in x]) <= slots
    for brand, g in df.groupby("brand"):
        prob += pulp.lpSum([x[s] for s in g.sku_id]) <= per_brand_cap
    for cat, g in df.groupby("category"):
        prob += pulp.lpSum([x[s] for s in g.sku_id]) >= min(per_cat_min, len(g))
    if budget is not None and price_map is not None:
        md_map = {}
        for _, r in df.iterrows():
            disc = price_map.get(r["sku_id"], 0.10)
            approx_md = r["base_price"]*disc
            md_map[r["sku_id"]] = approx_md
        prob += pulp.lpSum([x[s]*md_map[s] for s in x]) <= budget
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    chosen = [sid for sid, var in x.items() if var.value()==1]
    sol = df[df["sku_id"].isin(chosen)].copy()
    return sol, pulp.value(prob.objective), "OK"
def its_counterfactual(series:pd.Series, method="naive"):
    if method=="sarimax" and HAVE_SM and len(series)>30:
        try:
            model = SARIMAX(series, order=(1,1,1), seasonal_order=(0,1,1,52),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            pred = fit.get_prediction(start=0, end=len(series)-1)
            return pred.predicted_mean
        except Exception:
            pass
    return series.shift(52).fillna(series.mean())
# =========================
# DATA (AUTO-GENERATED)
# =========================
if "CAT" not in st.session_state:
    with st.spinner("Preparing synthetic retailer data (50 stores × 10 categories × 26 weeks)…"):
        catalog, stores, epos, WEEK_IDS = generate_epos()
        cvadi = compute_cv_adi(epos)
        st.session_state["CAT"] = catalog
        st.session_state["STORES"]= stores
        st.session_state["EPOS"] = epos
        st.session_state["WEEKS"] = WEEK_IDS
        st.session_state["CVADI"] = cvadi
        st.session_state["UPL_LAST"] = None
        st.session_state["HISTORY"] = []
catalog = st.session_state["CAT"]; stores = st.session_state["STORES"]
epos = st.session_state["EPOS"]; WEEK_IDS= st.session_state["WEEKS"]
cvadi = st.session_state["CVADI"]
# ---------- KPI HEADER ----------
k1,k2,k3,k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='kpi'><h5>Stores</h5><h2>{stores.shape[0]}</h2><small>Dubai · Abu Dhabi · Northern Emirates</small></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><h5>Categories</h5><h2>{len(CATS)}</h2><small>Weekly leaflet cadence</small></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><h5>SKUs</h5><h2>{catalog.shape[0]}</h2><small>Category → Subcategory → Brand → Tier</small></div>", unsafe_allow_html=True)
with k4:
    promorate = 100*epos["promo_flag"].mean()
    st.markdown(f"<div class='kpi'><h5>Promo Rate</h5><h2>{promorate:.1f}%</h2><small>Share of SKU-weeks on promo</small></div>", unsafe_allow_html=True)
st.markdown("<br/>", unsafe_allow_html=True)
# ---------- TOP LEVEL TABS ----------
top_tabs = st.tabs(["Executive Summary", "Approach", "App Demo", "Insights", "Recommendations"])
# =========================
# EXECUTIVE SUMMARY
# =========================
with top_tabs[0]:
    st.markdown("#### Answer-first: What to do this week")
    s1, s2 = st.columns([2,2])
    with s1:
        last = st.session_state.get("UPL_LAST", None)
        if last is None or len(last)==0:
            st.info("Run **Uplift Modeling** and **Leaflet Optimizer** to populate the weekly summary.")
        else:
            top10 = last.sort_values("pred_incremental_margin", ascending=False).head(10)
            pos_sum = last["pred_incremental_margin"].clip(lower=0).sum()
            capture = 100*top10["pred_incremental_margin"].clip(lower=0).sum()/max(1e-9, pos_sum)
            st.markdown(f"<div class='insight'>Top-10 SKUs capture ~{capture:.1f}% of predicted incremental margin. Concentrate slots there; validate ITS next week.</div>", unsafe_allow_html=True)
    with s2:
        st.markdown("<div class='helpbox'>This page auto-populates after you run <b>Uplift Modeling</b> and the <b>Leaflet Optimizer</b>. Use it in exec reviews.</div>", unsafe_allow_html=True)
# =========================
# APPROACH
# =========================
with top_tabs[1]:
    st.markdown("""
<div class="deck-card">
<h3>1) Goal & Business Problem</h3>
<div class="flowbox">
  <span class="flow-step">Goal: Maximize incremental margin</span><span class="arrow">→</span>
  <span class="flow-step">Decide: which SKUs & discounts</span><span class="arrow">→</span>
  <span class="flow-step">Operate: weekly leaflet by region</span>
</div>
<ul>
<li><b>Non-personalized, in-store</b> promos only.</li>
<li>Regions: Dubai, Abu Dhabi, Northern Emirates (or <b>All Regions</b> view).</li>
<li>We measure <b>incrementality</b> and <b>learn weekly</b>.</li>
</ul>
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div class="deck-card">
<h3>2) Situation → Complication → Resolution</h3>
<div class="flowbox">
  <span class="flow-step">Situation: Many promos, complex mix</span>
  <span class="arrow">⚡</span>
  <span class="flow-step">Complication: 20–50% ineffective / margin-negative</span>
  <span class="arrow">✅</span>
  <span class="flow-step">Resolution: Analytics-led choice of SKUs & depth</span>
</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div class="deck-card">
<h3>3) Data & Method Stack</h3>
<ul>
<li>EPOS weekly item sales (price, qty, promo/mechanic), product hierarchy with COGS, store master</li>
<li><b>Elasticity</b> → <b>Net-Impact</b> → <b>Uplift (T-learner)</b> → <b>ILP Leaflet</b> → <b>ITS</b> check → <b>Bandit</b> learning</li>
</ul>
</div>
""", unsafe_allow_html=True)
# =========================
# APP DEMO
# =========================
with top_tabs[2]:
    st.markdown("#### Guided demo")
    fc1, fc2 = st.columns([2,2])
    with fc1:
        region_global = st.selectbox("Global Region (applies where relevant)", REGION_OPTIONS, index=0, key="global_region")
    with fc2:
        cat_global = st.selectbox("Global Category (applies where relevant)", list(CATS.keys()), key="global_category")
    subtabs = st.tabs([
        "1) EDA", "2) Discount Tuner", "3) Uplift Modeling",
        "4) Leaflet Optimizer", "5) Impact Measurement",
        "6) Quality Assurance", "7) Learning Lab", "8) Help"
    ])
    # ----- 1) EDA -----
    with subtabs[0]:
        st.subheader("EDA: Where to play")
        st.markdown("<div class='helpbox'>Weekly granularity. Check promo intensity, demand stability, and margin pools before choosing SKUs.</div>", unsafe_allow_html=True)
        c1,c2 = st.columns([2,2])
        with c1:
            region = st.selectbox("Region", REGION_OPTIONS, index=REGION_OPTIONS.index(region_global) if region_global in REGION_OPTIONS else 0, key="eda_reg")
            category = st.selectbox("Category", list(CATS.keys()), index=list(CATS.keys()).index(cat_global), key="eda_cat")
        with c2:
            st.caption("Each chart shows a concise <b>Insight</b> with the business takeaway.", unsafe_allow_html=True)
        run_eda = st.button("Run EDA")
        if run_eda:
            s = _ensure_region(epos, region)
            s = s[s["category"]==category]
            promo_share = s.groupby("week_id")["promo_flag"].mean().reindex(WEEK_IDS).fillna(0)*100
            fig1 = go.Figure([go.Bar(x=WEEK_IDS, y=promo_share.values, name="% SKU-weeks on promo")])
            apply_dark(fig1, 300, f"Promo Intensity — {region} / {category}")
            fig1.update_xaxes(title="Week"); fig1.update_yaxes(title="% on promo")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown(f"<div class='insight'>Avg promo intensity last 8 weeks = <b>{promo_share[-8:].mean():.1f}%</b>. If high with flat margin, suspect fatigue or mis-targeting.</div>", unsafe_allow_html=True)
            cv_slice = cvadi.copy()
            if region != "All Regions":
                cv_slice = cv_slice[cv_slice["region"]==region]
            cv_slice = cv_slice.merge(catalog[["sku_id","category","brand","price_tier"]], on="sku_id", how="left")
            cv_slice = cv_slice[cv_slice["category"]==category]
            if len(cv_slice)>0:
                fig2 = px.scatter(cv_slice, x="cv", y="adi", color="price_tier", hover_data=["store_id","sku_id","brand"],
                                  title=f"Demand Stability vs Intermittency — {region} / {category}")
                apply_dark(fig2, 300)
                fig2.update_xaxes(title="CV (lower=stable)"); fig2.update_yaxes(title="ADI (lower=regular)")
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("<div class='insight'>Favor SKUs with <b>lower CV</b> (predictable lift) and <b>moderate ADI</b> (less pantry risk). Very high ADI suggests pull-forward.</div>", unsafe_allow_html=True)
            else:
                st.info("Not enough data for stability view.")
            c3,c4 = st.columns([2,2])
            with c3:
                h = s.groupby(["subcategory","brand","price_tier"]).agg(margin=("margin","sum")).reset_index()
                if len(h)>0:
                    fig3 = px.sunburst(h, path=["subcategory","brand","price_tier"], values="margin",
                                       title=f"Hierarchy Contribution (Margin) — {region} / {category}")
                    apply_dark(fig3, 380)
                    st.plotly_chart(fig3, use_container_width=True)
                    st.markdown("<div class='insight'>Hierarchy shows <b>breadth</b> vs <b>depth</b>. Avoid over-concentrating margin in one branch.</div>", unsafe_allow_html=True)
                else:
                    st.info("Not enough data for hierarchy view.")
            with c4:
                gm = s.groupby("sku_id").agg(gm=("gm%","mean")).reset_index()
                if len(gm)>0:
                    fig4 = px.histogram(gm, x="gm", nbins=30, title=f"Gross Margin % Distribution — {region} / {category}")
                    apply_dark(fig4, 380)
                    fig4.update_xaxes(title="GM%"); fig4.update_yaxes(title="SKU count")
                    st.plotly_chart(fig4, use_container_width=True)
                    st.markdown("<div class='insight'>Favor SKUs with healthy <b>GM%</b>; deep discounting low-GM% items risks margin dilution.</div>", unsafe_allow_html=True)
                else:
                    st.info("Not enough data for GM% distribution.")
    # ----- 2) Discount Tuner -----
    with subtabs[1]:
        st.subheader("Discount Tuner — Set the right discount")
        st.markdown("<div class='helpbox'>Estimate <b>elasticity</b> (log-log), compare <b>margin by discount</b>, then compute <b>Net Impact</b> (Lift + Halo − Cannibalization − Pull-forward) for a week/region.</div>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns([2,2,2])
        with c1:
            region_dt = st.selectbox("Region", REGION_OPTIONS, index=REGION_OPTIONS.index(region_global), key="dt_reg")
            cat_dt = st.selectbox("Category", list(CATS.keys()), index=list(CATS.keys()).index(cat_global), key="dt_cat")
            sku_list = catalog[catalog["category"]==cat_dt]["sku_id"].tolist()
            sku_dt = st.selectbox("SKU", sku_list, key="dt_sku")
        with c2:
            week_sel = st.slider("Week to analyze", 1, len(WEEK_IDS), len(WEEK_IDS), 1, key="dt_week")
            week_id = f"W{week_sel:03d}"
            disc_candidates = st.multiselect("Discount candidates", ["PCT10","PCT15","PCT20","BOGO"],
                                             default=["PCT10","PCT15","PCT20"], key="dt_disc")
        with c3:
            run_tuner = st.button("Compute Elasticity & Net-Impact", key="btn_tuner")
        if run_tuner:
            try:
                beta = quick_elasticity(epos, sku_dt, region_dt)
                if beta is not None:
                    st.markdown(
                        f"**Estimated elasticity (log-log)**: `{beta:.2f}` &nbsp;•&nbsp; "
                        f"<span class='helpbox'>Meaning: a 1% price drop changes demand by <b>{abs(beta):.2f}%</b>. "
                        f"{'More elastic' if abs(beta)>1 else 'Less elastic'} than 1.</span>", unsafe_allow_html=True)
                else:
                    st.info("Not enough data to estimate elasticity. We’ll use a category prior.")
                srow = catalog[catalog["sku_id"]==sku_dt].iloc[0]
                base_price = srow["base_price"]; cogs = srow["cogs"]
                hist = _ensure_region(epos, region_dt)
                hist = hist[(hist["sku_id"]==sku_dt)&(hist["promo_flag"]==0)].sort_values("date").tail(8)
                base_units = max(1, hist["qty"].mean() if len(hist)>0 else epos[epos["sku_id"]==sku_dt]["qty"].mean())
                xlab=[]; yval=[]
                for mech in disc_candidates:
                    d = MECHS[mech]
                    price = base_price*(1-d)
                    b = beta if beta is not None else -1.2
                    units = base_units*(price/base_price)**b
                    margin = (price-cogs)*units
                    xlab.append(mech); yval.append(margin)
                if yval:
                    fig = go.Figure([go.Bar(x=xlab, y=yval, text=[f"{v:,.0f}" for v in yval], textposition="auto")])
                    apply_dark(fig, 360, f"Estimated Margin by Discount — {region_dt} / {sku_dt}")
                    fig.update_xaxes(title="Mechanic"); fig.update_yaxes(title="Estimated Margin")
                    st.plotly_chart(fig, use_container_width=True)
                    best_mech = xlab[int(np.argmax(yval))]
                    st.markdown(f"<div class='insight'>**{best_mech}** maximizes margin under current elasticity & COGS. Deeper discounts aren’t always better.</div>", unsafe_allow_html=True)
                comps = net_impact_components(epos, sku_dt, week_id, region_dt)
                net = comps["lift"] + comps["halo"] - comps["cannib"] - comps["pull"]
                wf = go.Figure(go.Waterfall(
                    name="NetImpact", orientation="v",
                    measure=["relative","relative","relative","relative","total"],
                    x=["Lift","Halo","- Cannibalization","- Pull-forward","Net Incremental Margin"],
                    text=[f"{comps['lift']:.0f}", f"{comps['halo']:.0f}", f"-{comps['cannib']:.0f}", f"-{comps['pull']:.0f}", f"{net:.0f}"],
                    y=[comps['lift'], comps['halo'], -comps['cannib'], -comps['pull'], net]
                ))
                apply_dark(wf, 360, f"Net-Impact — {sku_dt} / {region_dt} / {week_id}")
                st.plotly_chart(wf, use_container_width=True)
                st.markdown(
                    f"<div class='insight'>Net incremental margin = <b>{net:,.0f}</b>. "
                    f"Drivers — Lift: <b>{comps['lift']:,.0f}</b>, Halo: <b>{comps['halo']:,.0f}</b>, "
                    f"Cannibalization: <b>-{comps['cannib']:,.0f}</b>, Pull-forward: <b>-{comps['pull']:,.0f}</b>.</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Discount tuner error: {e}")
    # ----- 3) Uplift Modeling -----
    with subtabs[2]:
        st.subheader("Uplift Modeling — Predict true incrementality (product level)")
        st.markdown("<div class='helpbox'>T-learner (with fallbacks) predicts <b>incremental margin</b> by SKU. The <b>Uplift Matrix</b> prioritises High baseline × High uplift. Shaded bands = CI.</div>", unsafe_allow_html=True)
        c1,c2 = st.columns([2,2])
        with c1:
            region_up = st.selectbox("Region", REGION_OPTIONS, index=REGION_OPTIONS.index(region_global), key="up_reg")
            category_up = st.selectbox("Category", list(CATS.keys()), index=list(CATS.keys()).index(cat_global), key="up_cat")
        with c2:
            recent_w = st.slider("Scoring window (recent weeks)", 4, 16, 8, 1)
            run_uplift = st.button("Run Uplift Models", key="btn_uplift")
        if run_uplift:
            with st.spinner("Training uplift models…"):
                upl = tlearner_uplift(epos, catalog, region=region_up, category=category_up, recent_weeks=recent_w)
            if upl.empty:
                st.info("Not enough data to fit uplift models. Try another region/category.")
            else:
                fig = px.scatter(upl, x="baseline_margin", y="pred_incremental_margin",
                                 color="price_tier", hover_data=["sku_id","brand","base_price","cogs","method"],
                                 title=f"Uplift Matrix — {region_up} / {category_up}")
                fig.update_traces(error_y=dict(
                    type='data',
                    array=(upl["upl_ci_high"]-upl["pred_incremental_margin"]).values,
                    arrayminus=(upl["pred_incremental_margin"]-upl["upl_ci_low"]).values,
                    visible=True
                ))
                apply_dark(fig, 440)
                fig.update_xaxes(title="Baseline Margin (recent non-promo)"); fig.update_yaxes(title="Predicted Incremental Margin")
                st.plotly_chart(fig, use_container_width=True)
                u_sorted = upl.sort_values("pred_incremental_margin", ascending=False).reset_index(drop=True)
                u_sorted["pos"] = u_sorted["pred_incremental_margin"].clip(lower=0)
                total = u_sorted["pos"].sum()
                u_sorted["cum_pct"] = np.where(total>0, 100*u_sorted["pos"].cumsum()/total, 0)
                figc = go.Figure()
                figc.add_trace(go.Scatter(y=u_sorted["cum_pct"], x=np.arange(1,len(u_sorted)+1),
                                          mode="lines+markers", name="Capture Curve"))
                apply_dark(figc, 320, f"Opportunity Capture Curve (Top-N SKUs) — {region_up} / {category_up}")
                figc.update_xaxes(title="Top-N SKUs"); figc.update_yaxes(title="% of predicted incremental captured")
                st.plotly_chart(figc, use_container_width=True)
                topn = u_sorted.head(10).merge(catalog, on="sku_id", suffixes=("","_c"))
                cov = (topn["pos"].sum() / max(1e-9, total))*100 if total>0 else 0
                st.markdown(f"<div class='insight'>Top 10 capture <b>{cov:.1f}%</b> of predicted opportunity. "
                            "Use this to pre-wire expectations with Category Managers.</div>", unsafe_allow_html=True)
                st.dataframe(topn[["sku_id","brand","category","base_price","cogs","pred_incremental_margin","upl_ci_low","upl_ci_high","baseline_margin","method"]], use_container_width=True)
                st.session_state["UPL_LAST"] = upl
                st.session_state["HISTORY"].append(
                    {"ts": datetime.utcnow().isoformat(timespec='seconds'), "region": region_up, "category": category_up,
                     "total_pos_uplift": float(u_sorted["pos"].sum()), "top10_capture_pct": float(cov)}
                )
    # ----- 4) Leaflet Optimizer -----
    with subtabs[3]:
        st.subheader("Leaflet Optimizer — ILP with guardrails")
        st.markdown("<div class='helpbox'>Objective: maximise Σ predicted <b>incremental margin</b>. Guardrails: <b>slots</b>, <b>per-category minimums</b>, <b>per-brand caps</b>, optional <b>markdown budget</b>. Discounts suggested via elasticity sweep.</div>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            slots = st.number_input("Leaflet slots", 6, 60, 24, 1)
            per_cat_min = st.number_input("Min per category", 0, 5, 1, 1)
        with c2:
            per_brand_cap = st.number_input("Max per brand", 1, 6, 3, 1)
            budget = st.number_input("Markdown budget (approx units)", 0, 200000, 0, 1000)
            budget = None if budget==0 else budget
        with c3:
            disc_for_budget = st.selectbox("Discount for budget approx", ["PCT10","PCT15","PCT20"], index=1)
            run_opt = st.button("Optimize Leaflet")
            st.caption("Run <b>Uplift Modeling</b> first; optimizer uses its results.", unsafe_allow_html=True)
        if run_opt:
            upl = st.session_state.get("UPL_LAST", None)
            if upl is None or upl.empty:
                st.warning("Run Uplift Models first.")
            else:
                price_map = {sid: MECHS[disc_for_budget] for sid in upl["sku_id"].unique()}
                sol, obj, status = ilp_leaflet(upl, slots=slots, per_cat_min=per_cat_min,
                                               per_brand_cap=per_brand_cap, budget=budget, price_map=price_map)
                if status!="OK":
                    st.error(status)
                else:
                    rec_disc=[]
                    for _, r in sol.iterrows():
                        best_d = 0.10; best_val=-1e9
                        for d in [0.10,0.15,0.20]:
                            b = quick_elasticity(epos, r["sku_id"], region_global)
                            b = b if b is not None else -1.2
                            base_price = r["base_price"]; cogs = r["cogs"]
                            base_units = max(1, epos[epos["sku_id"]==r["sku_id"]]["qty"].mean())
                            price = base_price*(1-d)
                            units = base_units*(price/r["base_price"])**b
                            margin = (price-cogs)*units
                            if margin>best_val: best_val=margin; best_d=d
                        rec_disc.append(best_d)
                    sol = sol.copy()
                    sol["rec_discount"] = rec_disc
                    sol["rec_price"] = (sol["base_price"]*(1-sol["rec_discount"])).round(2)
                    view = sol[["sku_id","brand","category","base_price","cogs","rec_discount","rec_price","pred_incremental_margin"]]
                    st.dataframe(view.sort_values("pred_incremental_margin", ascending=False), use_container_width=True)
                    cat_mix = view.groupby("category").size().reset_index(name="count")
                    figmix = px.bar(cat_mix, x="category", y="count", title=f"Leaflet Composition by Category — {region_global}")
                    apply_dark(figmix, 280)
                    st.plotly_chart(figmix, use_container_width=True)
                    share = view.groupby("category")["pred_incremental_margin"].sum().reset_index(name="incremental")
                    figshare = px.pie(share, values="incremental", names="category", title=f"Share of Predicted Incremental by Category — {region_global}")
                    apply_dark(figshare, 280)
                    st.plotly_chart(figshare, use_container_width=True)
                    st.success(f"Optimizer objective (Σ predicted incremental margin): {obj:,.0f}")
                    st.markdown("<div class='insight'>Use composition & share charts to communicate <b>portfolio balance</b> and where value concentrates. Adjust guardrails to change breadth vs depth.</div>", unsafe_allow_html=True)
                    st.download_button("Download Leaflet CSV", data=view.to_csv(index=False), file_name="leaflet_plan.csv", mime="text/csv")
    # ----- 5) Impact Measurement -----
    with subtabs[4]:
        st.subheader("Impact Measurement — ITS (business view)")
        st.markdown("<div class='helpbox'><b>ITS</b> builds a <b>counterfactual</b> from pre-period trend/seasonality. The <b>shaded area</b> after the change is measured incremental.</div>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns([2,2,2])
        with c1: region_eff = st.selectbox("Region", REGION_OPTIONS, index=REGION_OPTIONS.index(region_global), key="eff_reg")
        with c2: category_eff = st.selectbox("Category", list(CATS.keys()), index=list(CATS.keys()).index(cat_global), key="eff_cat")
        with c3: metric_eff = st.selectbox("Metric", ["revenue","margin","qty"], key="eff_met")
        run_its = st.button("Run Region×Category ITS")
        if run_its:
            sub = _ensure_region(epos, region_eff)
            s = sub[sub["category"]==category_eff].groupby("week_id")[metric_eff].sum().reindex(WEEK_IDS).fillna(0)
            cf = its_counterfactual(s, method="naive")
            inc = (s - cf)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=WEEK_IDS, y=s.values, mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=WEEK_IDS, y=cf.values, mode="lines", name="Counterfactual", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=WEEK_IDS, y=np.where(inc>0, s.values, cf.values),
                                     fill=None, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=WEEK_IDS, y=np.where(inc>0, cf.values, s.values),
                                     fill='tonexty', mode='lines', line=dict(width=0),
                                     name="Incremental Area"))
            apply_dark(fig, 420, f"{metric_eff.title()} — {region_eff} / {category_eff}")
            st.plotly_chart(fig, use_container_width=True)
            k1,k2,k3 = st.columns(3)
            with k1: st.metric("Cumulative Incremental", f"{inc.sum():,.0f}")
            with k2: st.metric("Peak Weekly Incremental", f"{inc.max():,.0f}")
            with k3: st.metric("Share Positive Weeks", f"{(inc>0).mean()*100:.1f}%")
            st.markdown("<div class='insight'>Before our change, counterfactual should track Actual closely — that’s the validity check. After the change, the shaded gap = value created.</div>", unsafe_allow_html=True)
    # ----- 6) Quality Assurance -----
    with subtabs[5]:
        st.subheader("Quality Assurance — trust the numbers")
        st.markdown("<div class='helpbox'>Auto-scan for issues that commonly break promo analytics. Fix upstream or exclude affected weeks.</div>", unsafe_allow_html=True)
        run_qa = st.button("Run QA Scan")
        if run_qa:
            grp = epos.sort_values("date").groupby(["store_id","sku_id"])
            def flag_unmarked(df):
                df = df.copy()
                df["price_prev"]=df["price"].shift(1)
                df["suspect_unflag"] = ((df["price_prev"]>0) & ((df["price_prev"]-df["price"])/df["price_prev"]>0.10) & (df["promo_flag"]==0)).astype(int)
                return df
            q = grp.apply(flag_unmarked).reset_index(drop=True)
            unflag_rate = 100*q["suspect_unflag"].mean()
            neg_margin = (epos["margin"]<0).mean()*100
            miss_cogs = epos["cogs"].isna().mean()*100
            k1,k2,k3 = st.columns(3)
            k1.markdown(f"<div class='kpi'><h5>Unflagged markdowns</h5><h2>{unflag_rate:.1f}%</h2><small>Price dropped >10% but promo flag=0</small></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='kpi'><h5>Negative margin rows</h5><h2>{neg_margin:.1f}%</h2><small>Check COGS & price integrity</small></div>", unsafe_allow_html=True)
            k3.markdown(f"<div class='kpi'><h5>Missing COGS</h5><h2>{miss_cogs:.1f}%</h2><small>Must be >0 for ROI math</small></div>", unsafe_allow_html=True)
            by_cat = q.groupby("category")["suspect_unflag"].mean().mul(100).reset_index(name="Unflag %")
            figq = px.bar(by_cat, x="category", y="Unflag %", title="Potential Unflagged Markdowns by Category")
            apply_dark(figq, 320)
            st.plotly_chart(figq, use_container_width=True)
            st.download_button("Download QA Findings (CSV)", data=by_cat.to_csv(index=False), file_name="qa_findings.csv")
    # ----- 7) Learning Lab -----
    with subtabs[6]:
        st.subheader("Learning Lab — weekly bandit on discount tiers")
        st.markdown("<div class='helpbox'>Simulate an ε-greedy bandit per category to learn which discount tier (10/15/20%) yields best <b>incremental margin</b>. This is how we keep improving.</div>", unsafe_allow_html=True)
        c1,c2 = st.columns([2,2])
        with c1:
            region_lb = st.selectbox("Region", REGION_OPTIONS, index=REGION_OPTIONS.index(region_global), key="lb_reg")
            category_lb = st.selectbox("Category", list(CATS.keys()), index=list(CATS.keys()).index(cat_global), key="lb_cat")
        with c2:
            weeks_sim = st.slider("Simulate next N weeks", 4, 12, 8, 1)
            eps = st.slider("Exploration ε", 0.0, 0.5, 0.15, 0.05)
        run_bandit = st.button("Run Learning Simulation")
        if run_bandit:
            upl = tlearner_uplift(epos, catalog, region=region_lb, category=category_lb)
            if upl.empty:
                st.info("Not enough data to derive priors. Try another region/category.")
            else:
                top = upl.sort_values("pred_incremental_margin", ascending=False).head(10).merge(catalog, on="sku_id", suffixes=("","_c"))
                means = {}
                for d in [0.10,0.15,0.20]:
                    gains=[]
                    for _, r in top.iterrows():
                        b = quick_elasticity(epos, r["sku_id"], region_lb)
                        b = b if b is not None else -1.2
                        price = r["base_price"]*(1-d)
                        base_units = max(1, epos[epos["sku_id"]==r["sku_id"]]["qty"].mean())
                        units = base_units*(price/r["base_price"])**b
                        margin = (price-r["cogs"])*units
                        gains.append(max(0, margin - (r["base_price"]-r["cogs"])*base_units))
                    means[d] = np.mean(gains)
                arms = [0.10,0.15,0.20]
                Q = {a: means.get(a,0) for a in arms}
                N = {a: 1 for a in arms}
                history = []
                for w in range(weeks_sim):
                    a = np.random.choice(arms) if np.random.rand()<eps else max(arms, key=lambda x: Q[x])
                    reward = np.random.normal(means[a], abs(means[a])*0.25 if means[a]!=0 else 1.0)
                    N[a]+=1; Q[a] += (reward - Q[a])/N[a]
                    history.append({"week": w+1, "discount": a, "reward": reward})
                hist = pd.DataFrame(history)
                figl = px.line(hist, x="week", y="reward", markers=True, title=f"Weekly Incremental Margin (Simulated) — {region_lb} / {category_lb}")
                apply_dark(figl, 320)
                figl.update_xaxes(title="Simulated Week"); figl.update_yaxes(title="Incremental Margin (approx)")
                st.plotly_chart(figl, use_container_width=True)
                share = hist["discount"].value_counts(normalize=True).sort_index().rename(lambda x: f"{int(x*100)}%").reset_index()
                share.columns=["Discount","Share"]
                figp = px.pie(share, values="Share", names="Discount", title=f"Discount Choices During Learning — {region_lb} / {category_lb}")
                apply_dark(figp, 280)
                figp.update_traces(textinfo="percent+label")
                st.plotly_chart(figp, use_container_width=True)
                st.markdown(f"<div class='insight'>Policy converges toward the best-performing tier. Exploration ε={eps} keeps testing alternatives to avoid getting stuck.</div>", unsafe_allow_html=True)
    # ----- 8) Help -----
    with subtabs[7]:
        st.subheader("Help & Glossary")
        st.markdown("""
- **Elasticity**: demand sensitivity to price. `−1.5` → 1% price drop ≈ 1.5% demand rise.
- **Net Impact**: Lift + Halo − Cannibalization − Pull-forward.
- **Uplift**: predicted incremental from promoting a SKU vs not promoting it.
- **ILP**: optimizer that chooses SKUs under constraints to maximize predicted value.
- **ITS**: counterfactual from pre-trend to estimate incremental after the change.
- **Bandit**: online learner balancing exploration and exploitation.
""")
        st.markdown("<div class='insight'>Lead with the <b>leaflet outcome</b> and <b>ITS impact</b>, then show the analytics trace (elasticity → uplift) as evidence.</div>", unsafe_allow_html=True)
# =========================
# INSIGHTS
# =========================
with top_tabs[3]:
    st.markdown("#### Auto-Insights")
    upl = st.session_state.get("UPL_LAST", None)
    if upl is None or upl.empty:
        st.info("Run **Uplift Modeling** to populate insights.")
    else:
        by_cat = upl.groupby("category")["pred_incremental_margin"].mean().sort_values(ascending=False)
        by_tier= upl.groupby("price_tier")["pred_incremental_margin"].mean().sort_values(ascending=False)
        bullets = []
        if len(by_cat)>0:
            bullets.append(f"Categories with highest average uplift: **{', '.join(by_cat.head(3).index.tolist())}**.")
        if len(by_tier)>0:
            bullets.append(f"Price tiers trend: **{by_tier.index[0]} ≥ {by_tier.index[-1]}** on uplift; tune discount depth accordingly.")
        top5 = upl.sort_values("pred_incremental_margin", ascending=False).head(5)["sku_id"].tolist()
        bullets.append(f"Top SKUs to validate via ITS next: **{', '.join(top5)}**.")
        st.markdown("<div class='deck-card'><h3>Key Patterns</h3><ul>"+ "".join([f"<li>{b}</li>" for b in bullets]) + "</ul></div>", unsafe_allow_html=True)
        hist = pd.DataFrame(st.session_state.get("HISTORY", []))
        if len(hist)>0:
            figt = px.line(hist, x="ts", y="total_pos_uplift", markers=True, title="Total Positive Uplift (Model) — Tracking")
            apply_dark(figt, 280)
            st.plotly_chart(figt, use_container_width=True)
            figc = px.line(hist, x="ts", y="top10_capture_pct", markers=True, title="Top-10 Capture % — Tracking")
            apply_dark(figc, 280)
            st.plotly_chart(figc, use_container_width=True)
            st.markdown("<div class='insight'>We should see <b>total positive uplift</b> rising week-on-week and <b>capture%</b> stabilizing as the optimizer converges.</div>", unsafe_allow_html=True)
# =========================
# RECOMMENDATIONS
# =========================
with top_tabs[4]:
    st.markdown("#### Recommendation Engine")
    upl = st.session_state.get("UPL_LAST", None)
    if upl is None or upl.empty:
        st.info("Run **Uplift Modeling** first to populate recommendations.")
    else:
        rec = upl.copy()
        rec["priority"] = rec["pred_incremental_margin"].clip(lower=0) * (1.0 - np.maximum(0, rec["pred_incremental_margin"]-rec["upl_ci_low"]) / (np.abs(rec["pred_incremental_margin"])+1e-9))
        rec = rec.sort_values(["priority","baseline_margin"], ascending=[False, False])
        st.markdown("<div class='deck-card'><h3>Playbook</h3><ol>"
                    "<li><b>Promote High-High:</b> high baseline × high uplift SKUs (top 10) — allocate slots first.</li>"
                    "<li><b>Probe Medium-High:</b> test 10% vs 15% with bandit to avoid over-discounting.</li>"
                    "<li><b>Protect GM:</b> avoid deep cuts on low-GM% or high cannibal risk categories.</li>"
                    "<li><b>Validate:</b> run ITS next week to quantify realized incremental and learn.</li>"
                    "</ol></div>", unsafe_allow_html=True)
        st.dataframe(rec[["sku_id","brand","category","price_tier","pred_incremental_margin","upl_ci_low","upl_ci_high","baseline_margin","priority","method"]].head(25),
                     use_container_width=True)
        st.download_button("Download Recommendations (CSV)",
                           data=rec.to_csv(index=False),
                           file_name="uplift_recommendations.csv",
                           mime="text/csv")

