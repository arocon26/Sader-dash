import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import hf_hub_download

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Sader Dash")

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- 2. LOGIN SCREEN CONFIGURATION ---
login_style = """
<style>
    .stApp { background-color: #f0f2f6; }
    [data-testid="stForm"] {
        background-color: #450084;
        padding: 50px;
        border-radius: 20px;
        box-shadow: 0px 10px 40px rgba(0,0,0,0.4);
        text-align: center;
        width: 600px !important;
        max-width: 90%;
        margin: 0 auto;
        display: block;
        margin-top: 100px;
    }
    .stTextInput input {
        background-color: white !important; color: black !important;
        border-radius: 10px; padding: 15px; font-size: 20px;
    }
    .stTextInput label {
        color: white !important; font-weight: bold; font-size: 24px; margin-bottom: 8px;
    }
    .stButton button {
        background-color: white !important; color: #450084 !important;
        font-weight: bold !important; font-size: 22px !important;
        border-radius: 10px !important; width: 100%; margin-top: 15px; border: none !important;
    }
    .stButton button:hover { background-color: #e0e0e0 !important; }
</style>
"""

def check_password():
    if st.session_state.get('password_correct', False):
        return True
    
    st.markdown(login_style, unsafe_allow_html=True)
    with st.form("login_form"):
        left, mid, right = st.columns([1, 2, 1])
        with mid:
            if os.path.exists("hc_logo.svg"):
                st.image("hc_logo.svg", use_container_width=True)
            elif os.path.exists("hc_logo.png"):
                st.image("hc_logo.png", use_container_width=True)
            else:
                st.markdown("<h1 style='text-align: center; color: white;'>⚾️</h1>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        password = st.text_input("Password:", type="password", key="password_input")
        if st.form_submit_button("LOG IN"):
            if password == "gocrossgo2026":
                st.session_state['password_correct'] = True
                st.rerun()
            else:
                st.error("❌ Incorrect Password")
    return False

if not check_password():
    st.stop()

# --- 3. GLOBAL CONSTANTS & PALETTES ---

# Statcast Standard Hex Codes
PITCH_PALETTE = {
    'Fastball': '#D22D49',  # Red
    'Four-Seam': '#D22D49',
    'Sinker': '#FE9D00',    # Orange
    'One-Seam Fastball': '#FE9D00',
    'Cutter': '#933F2C',    # Maroon
    'Slider': '#EEE716',    # Yellow
    'Sweeper': '#DDB33A',   # Gold
    'Curveball': '#00D1ED', # Cyan
    'Knuckle Curve': '#6236CD',
    'Changeup': '#1DBE3A',  # Green
    'Splitter': '#3BACAC',  # Teal
    'Knuckleball': '#867A08',
    'Other': '#999999'
}

NCAA_BASELINES = {
    'Fastball': {'Velo': 90.5, 'Spin': 2250, 'IVB': 16.0, 'HB': 8.0, 'Whiff': 18.0},
    'Sinker':   {'Velo': 89.5, 'Spin': 2100, 'IVB': 8.0,  'HB': 15.0, 'Whiff': 14.0},
    'Cutter':   {'Velo': 86.0, 'Spin': 2300, 'IVB': 6.0,  'HB': 2.0,  'Whiff': 26.0},
    'Slider':   {'Velo': 82.0, 'Spin': 2400, 'IVB': 2.0,  'HB': -12.0,'Whiff': 34.0},
    'Sweeper':  {'Velo': 79.0, 'Spin': 2500, 'IVB': 1.0,  'HB': -16.0,'Whiff': 38.0},
    'Curveball':{'Velo': 76.0, 'Spin': 2500, 'IVB': -8.0, 'HB': -10.0,'Whiff': 32.0},
    'Changeup': {'Velo': 81.0, 'Spin': 1600, 'IVB': 6.0,  'HB': 14.0, 'Whiff': 30.0},
}

ESSENTIAL_COLS = [
    'Pitcher', 'Batter', 'PitcherTeam', 'newestTeamName_Pitcher', 'newestTeamName_Batter', 'Date',
    'TaggedPitchType', 'RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 
    'RelHeight', 'RelSide', 'Extension', 'VertApprAngle', 'HorzApprAngle',
    'PlateLocHeight', 'PlateLocSide', 'PitchCall', 'KorBB', 'PlayResult', 'Balls', 'Strikes', 
    'PitchofPA', 'PAofInning', 'BatterSide', 'PitcherThrows',
    'ExitSpeed', 'Angle', 'Distance', 'Direction', 'run_value', 'wOBA_result', 'xwOBA_result',
    'Height_Pitcher', 'Weight_Pitcher', 'Jersey_Pitcher', 'Height_Batter', 'Weight_Batter', 'Jersey_Batter'
]

# --- 4. DATA LOADING FUNCTIONS ---

@st.cache_resource(show_spinner=False)
def get_local_data_path():
    """Cached helper to download/retrieve the parquet file path from Hugging Face Datasets"""
    return hf_hub_download(repo_id="arocon26/Sader-Data", repo_type="dataset", filename="ncaa_data_2025.parquet")

@st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
def load_team_names(app_type="hitter"):
    path = get_local_data_path()
    col = "newestTeamName_Batter" if app_type == "hitter" else "newestTeamName_Pitcher"
    try:
        df = pd.read_parquet(path, columns=[col])
        return sorted(df[col].dropna().unique())
    except Exception as e:
        return []

@st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
def load_players_for_team(team_name, app_type="hitter"):
    path = get_local_data_path()
    team_col = "newestTeamName_Batter" if app_type == "hitter" else "newestTeamName_Pitcher"
    player_col = "Batter" if app_type == "hitter" else "Pitcher"
    try:
        df = pd.read_parquet(path, columns=[team_col, player_col])
        df = df[df[team_col] == team_name]
        return sorted(df[player_col].dropna().unique())
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def load_player_data(player_name, team_name, app_type="hitter"):
    """Load ONLY data for selected player - Optimized with Smart Caching & Safe Index"""
    if app_type == "hitter":
        team_col = "newestTeamName_Batter"
        player_col = "Batter"
    else:
        team_col = "newestTeamName_Pitcher"
        player_col = "Pitcher"
    
    try:
        local_path = get_local_data_path()
        # Read FULL file first to preserve Master Index (Crucial for Admin Fixes)
        df = pd.read_parquet(local_path, columns=ESSENTIAL_COLS, engine='pyarrow')
        df = df[(df[team_col] == team_name) & (df[player_col] == player_name)]
        
        # Clean pitch types (But keep Sinker separate!)
        if not df.empty and 'TaggedPitchType' in df.columns:
            df['TaggedPitchType'] = df['TaggedPitchType'].replace({
                'ChangeUp': 'Changeup', 
                'One-Seam Fastball': 'Sinker'
            })
        return df
    except Exception:
        return pd.DataFrame()

# --- 5. INITIALIZE ADMIN STATE & HEADER ---
if "is_admin" not in st.session_state: st.session_state["is_admin"] = False
if "show_login" not in st.session_state: st.session_state["show_login"] = False

head_col1, head_col2 = st.columns([5, 1])
with head_col1:
    st.title("Sader Dash ✝️ ⚾️")
with head_col2:
    if not st.session_state["is_admin"]:
        if st.button("🔓 Edit Mode"): st.session_state["show_login"] = not st.session_state["show_login"]
    else:
        if st.button("🔒 Lock"):
            st.session_state["is_admin"] = False
            st.rerun()

if st.session_state["show_login"] and not st.session_state["is_admin"]:
    with st.sidebar:
        pwd = st.text_input("Admin Password", type="password")
        if pwd == "28HeritageHill":
            st.session_state["is_admin"] = True
            st.session_state["show_login"] = False
            st.rerun()

# --- 6. ALL HELPER FUNCTIONS ---

def draw_baseball_field():
    shapes = []
    shapes.append(dict(type="path", path="M 0 0 L 63.64 63.64 L 0 127.28 L -63.64 63.64 Z", line=dict(color="brown", width=2), fillcolor="rgba(139, 69, 19, 0.2)", layer="below"))
    shapes.append(dict(type="line", x0=0, y0=0, x1=233.3, y1=233.3, line=dict(color="white", width=3), layer="below"))
    shapes.append(dict(type="line", x0=0, y0=0, x1=-233.3, y1=233.3, line=dict(color="white", width=3), layer="below"))
    theta = np.linspace(-np.pi/4, np.pi/4, 100)
    fence_x, fence_y = 400 * np.sin(theta), 400 * np.cos(theta)
    path = f"M {fence_x[0]} {fence_y[0]}"
    for x, y in zip(fence_x[1:], fence_y[1:]): path += f" L {x} {y}"
    shapes.append(dict(type="path", path=path, line=dict(color="black", width=4), layer="below"))
    return shapes

def plot_trend_lines(df):
    """Plots metric trends over time (Game by Game)"""
    # 1. Prep Data & FORCE NUMERICS
    df = df.copy()
    
    # --- FIX: Ensure these columns are numbers before calculating mean ---
    numeric_cols = ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # -------------------------------------------------------------------

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # 2. Filter for significant pitch types (ignore types thrown < 5 times total)
    type_counts = df['TaggedPitchType'].value_counts()
    valid_types = type_counts[type_counts >= 5].index.tolist()
    df = df[df['TaggedPitchType'].isin(valid_types)]
    
    # 3. Group by Date
    daily = df.groupby(['Date', 'TaggedPitchType']).agg({
        'RelSpeed': 'mean',
        'SpinRate': 'mean',
        'InducedVertBreak': 'mean',
        'HorzBreak': 'mean'
    }).reset_index()
    
    # Sort by date
    daily = daily.sort_values('Date')
    
    # 4. Create Tabs
    t1, t2, t3, t4 = st.tabs(["Velocity", "Spin Rate", "Vertical Break", "Horizontal Break"])
    
    def create_trend_chart(data, y_col, title, y_label):
        fig = px.line(
            data, x='Date', y=y_col, color='TaggedPitchType',
            title=title, markers=True,
            color_discrete_map=PITCH_PALETTE,
            labels={y_col: y_label, 'Date': 'Date'}
        )
        fig.update_layout(height=450, plot_bgcolor='white', hovermode="x unified")
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        fig.update_xaxes(showgrid=True, gridcolor='whitesmoke')
        fig.update_yaxes(showgrid=True, gridcolor='whitesmoke')
        return fig
    
    with t1:
        st.plotly_chart(create_trend_chart(daily, 'RelSpeed', "Average Velocity Trend", "Velocity (MPH)"), use_container_width=True)
        
    with t2:
        st.plotly_chart(create_trend_chart(daily, 'SpinRate', "Average Spin Rate Trend", "Spin Rate (RPM)"), use_container_width=True)
        
    with t3:
        st.plotly_chart(create_trend_chart(daily, 'InducedVertBreak', "Vertical Break (IVB) Trend", "IVB (Inches)"), use_container_width=True)

    with t4:
        st.plotly_chart(create_trend_chart(daily, 'HorzBreak', "Horizontal Break Trend", "Horizontal Break (Inches)"), use_container_width=True)


def calc_zone_whiff_and_chase(df):
    df = df.copy()
    for c in ['PlateLocSide', 'PlateLocHeight', 'SzTop', 'SzBot']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    in_zone = (df['PlateLocSide'].between(-0.7083, 0.7083)) & (df['PlateLocHeight'].between(df.get('SzBot', 1.5), df.get('SzTop', 3.5)))
    swings = df['PitchCall'].str.lower().isin(['strikeswinging', 'foul', 'inplay'])
    whiffs = df['PitchCall'].str.lower() == 'strikeswinging'
    
    z_swing = (swings & in_zone).sum()
    zone_whiff_pct = ((whiffs & in_zone).sum() / z_swing * 100) if z_swing > 0 else 0.0
    
    o_pitch = (~in_zone).sum()
    chase_pct = ((swings & ~in_zone).sum() / o_pitch * 100) if o_pitch > 0 else 0.0
    return round(zone_whiff_pct, 1), round(chase_pct, 1)

def draw_tendency_heatmap(df):
    df = df.copy()
    for c in ['Distance', 'Direction']: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    valid_mask = df['PlayResult'].isin(['Single', 'Double', 'Triple', 'HomeRun', 'Out', 'Error', 'FieldersChoice']) & df['Distance'].notnull() & df['Direction'].notnull()
    df = df[valid_mask]
    total = len(df)
    if total == 0:
        st.warning("Not enough data for Heatmap."); return

    zones = {
        'L_IN': {'d_min': 0, 'd_max': 220, 'a_min': -45, 'a_max': -15, 'label': 'Pull (In)'},
        'C_IN': {'d_min': 0, 'd_max': 220, 'a_min': -15, 'a_max': 15, 'label': 'Center (In)'},
        'R_IN': {'d_min': 0, 'd_max': 220, 'a_min': 15, 'a_max': 45, 'label': 'Oppo (In)'},
        'L_OUT': {'d_min': 220, 'd_max': 400, 'a_min': -45, 'a_max': -15, 'label': 'Pull (Out)'},
        'C_OUT': {'d_min': 220, 'd_max': 400, 'a_min': -15, 'a_max': 15, 'label': 'Center (Out)'},
        'R_OUT': {'d_min': 220, 'd_max': 400, 'a_min': 15, 'a_max': 45, 'label': 'Oppo (Out)'}
    }
    
    fig = go.Figure()
    max_pct = 0
    results = {}
    for k, z in zones.items():
        count = len(df[(df['Distance'].between(z['d_min'], z['d_max'])) & (df['Direction'].between(z['a_min'], z['a_max']))])
        pct = count / total
        results[k] = pct
        if pct > max_pct: max_pct = pct

    for k, z in zones.items():
        pct = results[k]
        thetas = np.linspace(z['a_min'], z['a_max'], 20)
        r_outer, r_inner = z['d_max'], z['d_min']
        x = np.concatenate([r_outer * np.sin(np.deg2rad(thetas)), r_inner * np.sin(np.deg2rad(thetas[::-1]))])
        y = np.concatenate([r_outer * np.cos(np.deg2rad(thetas)), r_inner * np.cos(np.deg2rad(thetas[::-1]))])
        
        intensity = pct / max_pct if max_pct > 0 else 0
        gb = int(255 * (1 - intensity))
        fig.add_trace(go.Scatter(x=x, y=y, fill="toself", fillcolor=f'rgb(255, {gb}, {gb})', line=dict(color="black", width=1), showlegend=False, hoverinfo='text', text=f"{z['label']}: {pct:.1%}"))
        
        mid_theta, mid_r = (z['a_min'] + z['a_max']) / 2, (z['d_min'] + z['d_max']) / 2
        if z['d_min'] == 0: mid_r = z['d_max'] * 0.6
        fig.add_trace(go.Scatter(x=[mid_r * np.sin(np.deg2rad(mid_theta))], y=[mid_r * np.cos(np.deg2rad(mid_theta))], mode='text', text=[f"{pct:.0%}"], textfont=dict(size=14, color='black', weight='bold'), showlegend=False))

    fig.add_shape(type="line", x0=0, y0=0, x1=282, y1=282, line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=0, y0=0, x1=-282, y1=282, line=dict(color="black", width=2))
    fig.update_layout(title=dict(text="Hit Distribution %", x=0.5), width=400, height=400, xaxis=dict(visible=False, range=[-300, 300], scaleanchor="y"), yaxis=dict(visible=False, range=[-20, 420]), plot_bgcolor='white', margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def calc_pitcher_ab_stats(df):
    df = df.copy()
    for c in ['PlayResult', 'KorBB']: df[c] = df[c].fillna('Undefined').astype(str).str.lower().str.strip()
    
    hits = ['single', 'double', 'triple', 'homerun', 'home run']
    ab_res = hits + ['out', 'error', 'fielderschoice', 'fielders choice']
    ab_mask = (df['PlayResult'].isin(ab_res) | (df['KorBB'] == 'strikeout')) & ~df['PlayResult'].str.contains('sacrifice')
    ab_df = df[ab_mask]
    
    ab = len(ab_df)
    h = ab_df['PlayResult'].isin(hits).sum()
    bb = (df['KorBB'] == 'walk').sum()
    k = (df['KorBB'] == 'strikeout').sum()
    
    tb = (ab_df['PlayResult'] == 'single').sum() + (ab_df['PlayResult'] == 'double').sum()*2 + (ab_df['PlayResult'] == 'triple').sum()*3 + ab_df['PlayResult'].isin(['homerun', 'home run']).sum()*4
    avg = h/ab if ab>0 else 0
    slg = tb/ab if ab>0 else 0
    
    return {'AB': ab, 'H': h, 'AVG': round(avg, 3), 'SLG': round(slg, 3), 'BB': bb, 'K': k}

def get_advanced_metrics(df):
    """Calculate comprehensive batting metrics for any dataframe"""
    default = {"Pitches": 0, "AVG": 0, "OBP": 0, "SLG": 0, "OPS": 0, "wOBA": 0, "xwOBA": 0, "Avg EV": 0, "HardHit%": 0, "Whiff%": 0, "Chase%": 0, "K%": 0, "BB%": 0}
    if df.empty: return default
    
    df = df.copy()
    for col in ['ExitSpeed', 'Angle', 'PlateLocSide', 'PlateLocHeight', 'wOBA_result', 'xwOBA_result']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    res = df['PlayResult'].fillna('').str.lower()
    kbb = df['KorBB'].fillna('').str.lower()
    call = df['PitchCall'].fillna('').str.lower()
    swings = call.isin(['strikeswinging', 'foul', 'inplay'])
    misses = call == 'strikeswinging'
    in_zone = (df['PlateLocSide'].abs() <= 0.83) & (df['PlateLocHeight'].between(1.5, 3.5))
    
    hits = res.isin(['single', 'double', 'triple', 'homerun', 'home run']).sum()
    walks = (kbb == 'walk').sum()
    hbp = (call == 'hitbypitch').sum()
    abs_count = res.isin(['single', 'double', 'triple', 'homerun', 'home run', 'out', 'error', 'fielderschoice']).sum() + (kbb == 'strikeout').sum()
    pa_count = abs_count + walks + hbp + res.str.contains('sacrifice').sum()
    
    avg = hits / abs_count if abs_count > 0 else 0
    obp = (hits + walks + hbp) / pa_count if pa_count > 0 else 0
    tb = (res=='single').sum() + (res=='double').sum()*2 + (res=='triple').sum()*3 + res.isin(['homerun','home run']).sum()*4
    slg = tb / abs_count if abs_count > 0 else 0
    
    hh = (df['ExitSpeed'].dropna() >= 95).mean() * 100
    
    return {
        "Pitches": len(df), "AVG": avg, "OBP": obp, "SLG": slg, "OPS": obp + slg,
        "wOBA": df['wOBA_result'].mean(), "xwOBA": df['xwOBA_result'].mean(),
        "Avg EV": df['ExitSpeed'].mean(), 
        "Contact%": ((swings & ~misses).sum() / swings.sum() * 100) if swings.sum() > 0 else 0,
        "Swing%": (swings.sum() / len(df) * 100) if len(df) > 0 else 0,
        "Chase%": ((swings & ~in_zone).sum() / (~in_zone).sum() * 100) if (~in_zone).sum() > 0 else 0,
        "Miss%": (misses.sum() / swings.sum() * 100) if swings.sum() > 0 else 0,
        "K%": ((kbb == 'strikeout').sum() / pa_count * 100) if pa_count > 0 else 0,
        "BB%": (walks / pa_count * 100) if pa_count > 0 else 0,
        "90th EV": np.percentile(df['ExitSpeed'].dropna(), 90) if not df['ExitSpeed'].dropna().empty else 0,
        "HardHit%": hh,
        "LD%": df['Angle'].dropna().between(10, 25).mean() * 100,
        "GB%": (df['Angle'].dropna() < 10).mean() * 100,
        "FB%": df['Angle'].dropna().between(25, 50).mean() * 100,
        "Z-Contact%": ((swings & in_zone & ~misses).sum() / (swings & in_zone).sum() * 100) if (swings & in_zone).sum() > 0 else 0,
        "HHLD%": ((df['ExitSpeed'] >= 95) & (df['Angle'].between(8, 32))).sum() / len(df.dropna(subset=['ExitSpeed'])) * 100 if len(df.dropna(subset=['ExitSpeed'])) > 0 else 0
    }

def overall_stats(df):
    m = get_advanced_metrics(df)
    zone_whiff, chase = calc_zone_whiff_and_chase(df)
    # Return specific dict for table formatting
    return {
        'Pitch': 'Overall', '#': len(df), 'Usage': 100.0,
        'AVG': round(m['AVG'], 3), 'SLG': round(m['SLG'], 3),
        'Zone%': 0, # Calc if needed
        'Whiff%': round(m['Miss%'], 1), 'HH%': round(m['HardHit%'], 1),
        'GB%': round(m['GB%'], 1), 'Zone Whiff%': round(zone_whiff, 1),
        'Chase%': round(chase, 1), 'run_value': 0, 
        'wOBA': round(m['wOBA'], 3), 'xwOBA': round(m['xwOBA'], 3)
    }

def pitch_type_stats(df):
    results = []
    total = len(df)
    for pitch, group in df.groupby('TaggedPitchType'):
        m = get_advanced_metrics(group)
        z_whiff, chase = calc_zone_whiff_and_chase(group)
        rv = pd.to_numeric(group['run_value'], errors='coerce').mean()
        usage = len(group) / total * 100
        
        results.append({
            'Pitch': pitch, '#': len(group), 'Usage': round(usage, 1),
            'AVG': round(m['AVG'], 3), 'SLG': round(m['SLG'], 3),
            'Zone%': 0, 'Whiff%': round(m['Miss%'], 1),
            'HH%': round(m['HardHit%'], 1), 'GB%': round(m['GB%'], 1),
            'Zone Whiff%': round(z_whiff, 1), 'Chase%': round(chase, 1),
            'run_value': round(rv, 1), 'wOBA': round(m['wOBA'], 1), 'xwOBA': round(m['xwOBA'], 1)
        })
    return pd.DataFrame(results)

def draw_performance_grid(data, hand):
    """Draws the 2x2 strike zone grid with stats."""
    hand_col = 'PitcherThrows' if 'PitcherThrows' in data.columns else 'PitcherHand'
    df = data[data[hand_col] == hand].copy() if hand_col in data.columns else data.copy()

    df['PlateLocSide'] = pd.to_numeric(df['PlateLocSide'], errors='coerce')
    df['PlateLocHeight'] = pd.to_numeric(df['PlateLocHeight'], errors='coerce')
    df['Zone'] = 'Outside'
    df.loc[(df['PlateLocSide'] < 0) & (df['PlateLocHeight'] > 2.5), 'Zone'] = 'Upper Left'
    df.loc[(df['PlateLocSide'] >= 0) & (df['PlateLocHeight'] > 2.5), 'Zone'] = 'Upper Right'
    df.loc[(df['PlateLocSide'] < 0) & (df['PlateLocHeight'] <= 2.5), 'Zone'] = 'Lower Left'
    df.loc[(df['PlateLocSide'] >= 0) & (df['PlateLocHeight'] <= 2.5), 'Zone'] = 'Lower Right'

    fig_zone = go.Figure()
    quads = [[[-0.83, 2.5, 0, 3.5], 'Upper Left'], [[0, 2.5, 0.83, 3.5], 'Upper Right'],
             [[-0.83, 1.5, 0, 2.5], 'Lower Left'], [[0, 1.5, 0.83, 2.5], 'Lower Right']]

    for coords, name in quads:
        z_df = df[df['Zone'] == name]
        if not z_df.empty:
            m = get_advanced_metrics(z_df)
            bg_color = "rgba(34, 139, 34, 0.9)" if m['OPS'] >= 0.800 else "rgba(178, 34, 34, 0.9)"
            fig_zone.add_shape(type="rect", x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3],
                line=dict(color="white", width=1), fillcolor=bg_color, layer="below")
            fig_zone.add_trace(go.Scatter(
                x=[(coords[0]+coords[2])/2], y=[(coords[1]+coords[3])/2],
                text=f"OPS: {m['OPS']:.3f}<br>AVG: {m['AVG']:.3f}<br>EV: {m['Avg EV']:.1f}",
                mode="text", textfont=dict(size=14, color="white", weight="bold"), showlegend=False
            ))
        else:
            fig_zone.add_shape(type="rect", x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3],
                line=dict(color="gray", width=1), fillcolor="rgba(100, 100, 100, 0.3)", layer="below")

    fig_zone.add_shape(type="path", path="M -0.4 0 L 0.4 0 L 0.4 0.2 L 0 0.4 L -0.4 0.2 Z",
                  line=dict(color="white", width=2), fillcolor="gray", layer="below")
    fig_zone.update_layout(
        title=dict(text=f"vs {hand}HP", x=0.5, font=dict(size=16, color="white")),
        xaxis=dict(range=[-1.5, 1.5], visible=False), yaxis=dict(range=[-0.5, 4.0], visible=False), 
        width=350, height=450, margin=dict(l=10, r=10, t=40, b=10), 
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig_zone



app = st.selectbox("Select App", ["Home", "NCAA Hitter", "NCAA Pitcher", "Holy Cross Hitter", "Holy Cross Pitcher"])

if app == "Home":
    st.write("## Welcome to Sader Dash! ⚾️ ✝️")
    st.write("Select an app from the dropdown above to get started.")
    st.write("- **NCAA Hitter**: View stats for any NCAA baseball hitter")
    st.write("- **NCAA Pitcher**: Analyze any NCAA pitcher's performance")
    st.write("- **Holy Cross Hitter**: Quick access to Holy Cross batters")
    st.write("- **Holy Cross Pitcher**: Quick access to Holy Cross pitchers")
    
    st.info("💡 **Tip**: Data will be downloaded automatically when you select your first player (one-time ~30 second download, then cached).")
    
    # Stop execution here - no data loading
    st.stop()

elif app == "NCAA Hitter":
    st.subheader("⚾ NCAA Hitter Stats")
    # ... rest of your code

    # Step 1: Load ONLY team names (fast!)
    teams = load_team_names(app_type="hitter")
    selected_team_full = st.sidebar.selectbox("Select a Team", options=teams)
    
    # Step 2: Load ONLY player names for selected team
    batters = load_players_for_team(selected_team_full, app_type="hitter")
    batters_formatted = [' '.join(b.split(', ')[::-1]) if ', ' in b else b for b in batters]
    selected_batter_fmt = st.sidebar.selectbox("Select Batter", options=batters_formatted)
    
    # Convert back to raw format
    selected_batter_raw = ', '.join(selected_batter_fmt.split(' ')[::-1]) if ' ' in selected_batter_fmt else selected_batter_fmt

    # Step 3: Load ONLY this player's data (lazy loaded!)
    player_data = load_player_data(selected_batter_raw, selected_team_full, app_type="hitter")
    
    # --- PLAYER BIO HEADER ---
    if not player_data.empty:
        p_info = player_data.iloc[0]
        p_height = p_info['Height_Batter'] if pd.notnull(p_info['Height_Batter']) else "N/A"
        p_weight = f"{int(p_info['Weight_Batter'])} lbs" if pd.notnull(p_info['Weight_Batter']) else ""
        p_jersey = f"#{int(p_info['Jersey_Batter'])}" if pd.notnull(p_info['Jersey_Batter']) else ""
        
        batter_side = p_info['BatterSide']
        hand_display = "LHH" if str(batter_side).upper().startswith("L") else "RHH" if str(batter_side).upper().startswith("R") else ""

        st.header(f"{selected_batter_fmt} {p_jersey}")
        st.subheader(f"{selected_team_full} | {hand_display}")
        st.write(f"**Physical Profile:** {p_height} | {p_weight}")
        st.divider()
    
    # ... rest of your NCAA Hitter code stays the same ...



    # --- CALCS ---
    overall = get_advanced_metrics(player_data)
    lhp_data = player_data[player_data["PitcherThrows"].str.contains('L|l', na=False)]
    rhp_data = player_data[player_data["PitcherThrows"].str.contains('R|r', na=False)]
    l_m = get_advanced_metrics(lhp_data)
    r_m = get_advanced_metrics(rhp_data)

    tab1, tab2, tab3 = st.tabs(["Stats Overview", "Spray Chart", "Heat Maps"])

    with tab1:
        # --- 1. Overall Production Table ---
        st.markdown("#### Overall Production")
        overall_df = pd.DataFrame({
            "AVG": [f"{overall['AVG']:.3f}"], 
            "OBP": [f"{overall['OBP']:.3f}"], 
            "SLG": [f"{overall['SLG']:.3f}"], 
            "wOBA": [f"{overall['wOBA']:.3f}"], 
            "xwOBA": [f"{overall['xwOBA']:.3f}"]
        })
        st.table(overall_df)
        
        # --- 2. Contact & Plate Discipline Table ---
        st.markdown("#### Contact & Plate Discipline")
        discipline_df = pd.DataFrame({
            "90th EV": [f"{overall['90th EV']:.1f}"],
            "Hard Hit%": [f"{overall['HardHit%']:.1f}%"],
            "BB%": [f"{overall['BB%']:.1f}%"],
            "K%": [f"{overall['K%']:.1f}%"],
            "Chase%": [f"{overall['Chase%']:.1f}%"],
            "Z-Contact%": [f"{overall['Z-Contact%']:.1f}%"],
            "LD%": [f"{overall['LD%']:.1f}%"],
            "GB%": [f"{overall['GB%']:.1f}%"],
            "FB%": [f"{overall['FB%']:.1f}%"],
            "HHLD%": [f"{overall['HHLD%']:.1f}%"]
        })
        st.table(discipline_df)

        st.divider()

        # --- 3. Split Performance Logic ---
        
        def get_pitch_group_splits(df):
            """Categorizes pitches and calculates metrics for each group."""
            fastballs = ['Fastball', 'Sinker', 'FourSeamFastBall', 'TwoSeamFastBall', 'Cutter']
            breaking = ['Slider', 'Curveball', 'Sweeper', 'KnuckleCurve']
            offspeed = ['ChangeUp', 'Splitter', 'Knuckleball']
            
            df = df.copy()
            df['Category'] = 'Other'
            df.loc[df['TaggedPitchType'].isin(fastballs), 'Category'] = 'Fastballs'
            df.loc[df['TaggedPitchType'].isin(breaking), 'Category'] = 'Breaking'
            df.loc[df['TaggedPitchType'].isin(offspeed), 'Category'] = 'Off-Speed'
            
            rows = []
            # Calculate metrics for Overall split and each sub-category
            rows.append({"Type": "OVERALL", **get_advanced_metrics(df)})
            for cat in ['Fastballs', 'Breaking', 'Off-Speed']:
                cat_df = df[df['Category'] == cat]
                if not cat_df.empty:
                    rows.append({"Type": cat.upper(), **get_advanced_metrics(cat_df)})
            
            res_df = pd.DataFrame(rows)
            # Format columns for display
            formatted = pd.DataFrame({
                "Pitch Type": res_df['Type'],
                "Pitches": res_df['Pitches'],
                "Avg EV": res_df['Avg EV'].map("{:.1f}".format),
                "Avg LA": res_df['Avg LA'].map("{:.1f}°".format),
                "OPS": res_df['OPS'].map("{:.3f}".format),
                "wOBA": res_df['wOBA'].map("{:.3f}".format),
                "HardHit%": res_df['HardHit%'].map("{:.1f}%".format),
                "HHLD%": res_df['HHLD%'].map("{:.1f}%".format),
                "Contact%": res_df['Contact%'].map("{:.1f}%".format),
                "Whiff%": res_df['Miss%'].map("{:.1f}%".format),
                "Chase%": res_df['Chase%'].map("{:.1f}%".format),
                "K%": res_df['K%'].map("{:.1f}%".format)
            })
            return formatted

        # --- vs LHP Performance ---
        st.markdown("#### vs Left-Handed Pitchers")
        if not lhp_data.empty:
            lhp_split_df = get_pitch_group_splits(lhp_data)
            st.dataframe(lhp_split_df.style.hide(axis='index'), use_container_width=True)
        else:
            st.info("No data available vs Left-Handed Pitchers")

        # --- vs RHP Performance ---
        st.markdown("#### vs Right-Handed Pitchers")
        if not rhp_data.empty:
            rhp_split_df = get_pitch_group_splits(rhp_data)
            st.dataframe(rhp_split_df.style.hide(axis='index'), use_container_width=True)
        else:
            st.info("No data available vs Right-Handed Pitchers")

    with tab2:
        st.subheader(f"Spray Chart Analysis: {selected_batter_fmt}")
        
        # 1. Prepare and Clean Data
        spray_df = player_data.copy()
        spray_df['Distance'] = pd.to_numeric(spray_df['Distance'], errors='coerce')
        spray_df['Direction'] = pd.to_numeric(spray_df['Direction'], errors='coerce')
        spray_df['ExitSpeed'] = pd.to_numeric(spray_df['ExitSpeed'], errors='coerce')
        
        # Filter: Exclude undefined and foul results
        to_exclude = ['undefined', 'foul', 'null', 'nan']
        spray_df = spray_df[~spray_df['PlayResult'].astype(str).str.lower().isin(to_exclude)]
        spray_df = spray_df.dropna(subset=['Distance', 'Direction'])

        if not spray_df.empty:
            # Create two columns layout
            col_spray, col_heat = st.columns([1.5, 1])
            
            with col_spray:
                # 2. Coordinate Calculation
                spray_df['hc_x'] = spray_df['Distance'] * np.sin(np.deg2rad(spray_df['Direction']))
                spray_df['hc_y'] = spray_df['Distance'] * np.cos(np.deg2rad(spray_df['Direction']))
                
                fig = go.Figure()

                # Custom Color Palette
                color_map = {
                    'single': '#FFD700', 'double': '#00CD66', 'triple': '#00F5FF', 
                    'homerun': '#9370DB', 'home run': '#9370DB',
                    'sacrifice': '#708090', 'fielderschoice': '#708090', 
                    'error': '#708090', 'out': '#708090'
                }
                
                # 3. Add Data Traces
                for res, group in spray_df.groupby('PlayResult'):
                    res_key = res.lower().replace(" ", "").strip()
                    fig.add_trace(go.Scatter(
                        x=group['hc_x'], y=group['hc_y'], mode='markers', name=res.title(),
                        customdata=np.stack((group['PlayResult'], group['ExitSpeed'], group['Distance']), axis=-1),
                        marker=dict(
                            size=10, 
                            color=color_map.get(res_key, '#708090'), 
                            opacity=0.8, 
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate="<b>Result:</b> %{customdata[0]}<br><b>EV:</b> %{customdata[1]:.1f} mph<br><b>Dist:</b> %{customdata[2]:.0f} ft<extra></extra>"
                    ))
                
                # 4. Field Layout
                fig.update_layout(
                    shapes=draw_baseball_field(),
                    yaxis=dict(scaleanchor="x", scaleratio=1, visible=False, range=[-20, 450]),
                    xaxis=dict(visible=False, range=[-300, 300]),
                    width=700, height=650, plot_bgcolor='white',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_heat:
                st.markdown("#### 🎯 Hit Tendency")
                # CALL THE HELPER FUNCTION HERE
                draw_tendency_heatmap(spray_df)
                
                # Quick Text Summary
                pull_ev = spray_df[spray_df['Direction'] < -15]['ExitSpeed'].mean()
                oppo_ev = spray_df[spray_df['Direction'] > 15]['ExitSpeed'].mean()
                st.metric("Avg EV (Pull)", f"{pull_ev:.1f} mph")
                st.metric("Avg EV (Oppo)", f"{oppo_ev:.1f} mph")

        else:
            st.warning("No valid fair-ball data available.")

        

    with tab3:
        st.subheader("Launch Angle vs. Exit Velocity (Airborne Fair Balls)")
        
        # --- 1. DATA PREP FOR SCATTER PLOT ---
        plot_df = player_data.copy()
        plot_df['ExitSpeed'] = pd.to_numeric(plot_df['ExitSpeed'], errors='coerce')
        plot_df['Angle'] = pd.to_numeric(plot_df['Angle'], errors='coerce')
        
        plot_df['PlayResult_Lower'] = plot_df['PlayResult'].fillna('').astype(str).str.lower().str.strip()
        
        to_exclude = ['undefined', 'foul', 'null', 'nan', '']
        plot_df = plot_df[
            (plot_df['Angle'] >= 0) & 
            (~plot_df['PlayResult_Lower'].isin(to_exclude))
        ].copy()
        
        plot_df = plot_df.dropna(subset=['ExitSpeed', 'Angle'])

        # --- 2. SCATTER PLOT & PERFORMANCE GRID ---
        if not plot_df.empty:
            fig = px.scatter(
                plot_df, x='ExitSpeed', y='Angle', color='PlayResult_Lower',
                labels={'ExitSpeed': 'Exit Velocity (mph)', 'Angle': 'Launch Angle (deg)', 'PlayResult_Lower': 'Result'},
                color_discrete_map={
                    'single': '#FFD700', 'double': '#00CD66', 'triple': '#00F5FF', 
                    'homerun': '#9370DB', 'home run': '#9370DB', 'sacrifice': '#708090', 
                    'fielderschoice': '#708090', 'error': '#708090', 'out': '#708090'
                },
                hover_data={'ExitSpeed': ':.1f', 'Angle': ':.1f', 'PlayResult': True}
            )
            fig.add_hrect(y0=10, y1=30, line_width=0, fillcolor="red", opacity=0.1, 
                            annotation_text="Power Alley (10°-30°)", annotation_position="top left")
            fig.add_vline(x=95, line_dash="dash", line_color="black", annotation_text="Hard Hit (95+)")
            fig.update_layout(width=800, height=600, plot_bgcolor='white', legend_title_text='Play Result')
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            st.subheader("📊 Performance by Zone")
            
            def draw_performance_grid(data, hand):
                hand_col = 'PitcherThrows' if 'PitcherThrows' in data.columns else 'PitcherHand'
                
                df = data[data[hand_col] == hand].copy()
                df['PlateLocSide'] = pd.to_numeric(df['PlateLocSide'], errors='coerce')
                df['PlateLocHeight'] = pd.to_numeric(df['PlateLocHeight'], errors='coerce')
                
                # Define Zones
                df['Zone'] = 'Outside'
                df.loc[(df['PlateLocSide'] < 0) & (df['PlateLocHeight'] > 2.5), 'Zone'] = 'Upper Left'
                df.loc[(df['PlateLocSide'] >= 0) & (df['PlateLocHeight'] > 2.5), 'Zone'] = 'Upper Right'
                df.loc[(df['PlateLocSide'] < 0) & (df['PlateLocHeight'] <= 2.5), 'Zone'] = 'Lower Left'
                df.loc[(df['PlateLocSide'] >= 0) & (df['PlateLocHeight'] <= 2.5), 'Zone'] = 'Lower Right'

                fig_zone = go.Figure()
                
                # Coordinates for the 4 zones [x0, y0, x1, y1]
                quads = [[[-0.83, 2.5, 0, 3.5], 'Upper Left'], [[0, 2.5, 0.83, 3.5], 'Upper Right'],
                         [[-0.83, 1.5, 0, 2.5], 'Lower Left'], [[0, 1.5, 0.83, 2.5], 'Lower Right']]

                for coords, name in quads:
                    z_df = df[df['Zone'] == name]
                    if not z_df.empty:
                        m = get_advanced_metrics(z_df)
                        
                        # COLOR LOGIC: Darker Red/Green so White text is readable
                        # Green = Forest Green (High Opacity)
                        # Red = Firebrick (High Opacity)
                        bg_color = "rgba(34, 139, 34, 0.9)" if m['OPS'] >= 0.800 else "rgba(178, 34, 34, 0.9)"
                        
                        # 1. Draw the Colored Rectangle (Layer=Below ensures text sits on top)
                        fig_zone.add_shape(type="rect", 
                            x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3],
                            line=dict(color="white", width=1), 
                            fillcolor=bg_color,
                            layer="below" 
                        )
                        
                        # 2. Draw the Text (Bright White & Bold)
                        fig_zone.add_trace(go.Scatter(
                            x=[(coords[0]+coords[2])/2], y=[(coords[1]+coords[3])/2],
                            text=f"OPS: {m['OPS']:.3f}<br>AVG: {m['AVG']:.3f}<br>EV: {m['Avg EV']:.1f}",
                            mode="text", 
                            textfont=dict(size=14, color="white", weight="bold"), # <--- THE KEY FIX
                            showlegend=False
                        ))
                    else:
                        # Empty Zone (Gray)
                        fig_zone.add_shape(type="rect", 
                            x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3],
                            line=dict(color="gray", width=1), 
                            fillcolor="rgba(100, 100, 100, 0.3)",
                            layer="below"
                        )

                # Draw Home Plate
                fig_zone.add_shape(type="path", path="M -0.4 0 L 0.4 0 L 0.4 0.2 L 0 0.4 L -0.4 0.2 Z",
                              line=dict(color="white", width=2), fillcolor="gray", layer="below")

                fig_zone.update_layout(
                    title=dict(text=f"vs {hand}HP", x=0.5, font=dict(size=16, color="white")),
                    xaxis=dict(range=[-1.5, 1.5], visible=False),
                    yaxis=dict(range=[-0.5, 4.0], visible=False), 
                    width=350, height=450,
                    margin=dict(l=10, r=10, t=40, b=10), 
                    plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                return fig_zone

            z_col1, z_col2 = st.columns(2)
            with z_col1:
                st.plotly_chart(draw_performance_grid(player_data, 'Left'), use_container_width=True)
            with z_col2:
                st.plotly_chart(draw_performance_grid(player_data, 'Right'), use_container_width=True)

        else:
            st.warning("No airborne contact data available.")

        st.divider()

        # --- 3. ADVANCED HEATMAPS (Runs independently of Scatter Plot) ---
        st.subheader("🔥 Zone Heatmaps")
        st.caption("Pitch density visualization by pitch family.")

        # Imports
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Prepare Data
        df_heat = player_data.copy()

        # Force numeric types
        for col in ["PlateLocSide", "PlateLocHeight", "ExitSpeed"]:
            df_heat[col] = pd.to_numeric(df_heat[col], errors='coerce')

        # Drop invalid rows
        df_heat = df_heat.dropna(subset=["PlateLocSide", "PlateLocHeight"])

        # Define Pitch Families
        def get_pitch_family(pitch_type):
            pitch_type = str(pitch_type).lower()
            if any(x in pitch_type for x in ['fastball', 'sinker', 'cutter', 'four', 'seam']): return 'Fastballs'
            if any(x in pitch_type for x in ['slider', 'curve', 'sweeper', 'slurve', 'knucklecurve']): return 'Breaking'
            if any(x in pitch_type for x in ['change', 'split', 'knuckle', 'fork']): return 'Off-Speed'
            return 'Other'

        df_heat['PitchFamily'] = df_heat['TaggedPitchType'].apply(get_pitch_family)

        # Filters
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            map_types = ["All Pitches", "Whiffs", "Hard Hit (95+)", "Softly Hit (<80)", "Chases", "Called Strikes"]
            map_sel = st.selectbox("Select Metric", map_types, key="hmap_metric_hitter_tab3")
        
        with h_col2:
            pitcher_side_sel = st.radio("Pitcher Throws", ["Combined", "Right", "Left"], horizontal=True, key="hmap_side_hitter_tab3")

        # Apply Pitcher Hand Filter
        if pitcher_side_sel == "Right":
            df_heat = df_heat[df_heat["PitcherThrows"] == "Right"]
        elif pitcher_side_sel == "Left":
            df_heat = df_heat[df_heat["PitcherThrows"] == "Left"]

        # Apply Metric Filter
        if map_sel == "Whiffs":
            df_event = df_heat[df_heat["PitchCall"] == "StrikeSwinging"]
        elif map_sel == "Hard Hit (95+)":
            df_event = df_heat[df_heat["ExitSpeed"] >= 95]
        elif map_sel == "Softly Hit (<80)":
            df_event = df_heat[df_heat["ExitSpeed"] <= 80]
        elif map_sel == "Chases":
            swing_calls = ["strikeswinging", "foul", "inplay"]
            in_zone = df_heat["PlateLocSide"].between(-0.83, 0.83) & df_heat["PlateLocHeight"].between(1.5, 3.5)
            df_event = df_heat[df_heat["PitchCall"].str.lower().isin(swing_calls) & ~in_zone]
        elif map_sel == "Called Strikes":
            df_event = df_heat[df_heat["PitchCall"] == "StrikeCalled"]
        else: # All Pitches
            df_event = df_heat

        # Render Heatmaps
        if df_event.empty:
            st.warning(f"No data found for {map_sel} vs {pitcher_side_sel} Handed Pitchers.")
        else:
            families = ['Fastballs', 'Breaking', 'Off-Speed']
            cols = st.columns(3)

            for i, family in enumerate(families):
                with cols[i]:
                    subset = df_event[df_event['PitchFamily'] == family]
                    
                    fig, ax = plt.subplots(figsize=(4, 5))
                    
                    ax.add_patch(Rectangle((-0.83, 1.5), 1.66, 2.0, 
                                        fill=False, edgecolor="black", linewidth=2.5, zorder=10))
                    ax.plot([-0.83, 0.83, 0.83, 0, -0.83, -0.83], [0, 0, 0.15, 0.3, 0.15, 0], color="black", lw=1.5)

                    if len(subset) < 5:
                        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, fontsize=12, color='gray')
                    else:
                        try:
                            sns.kdeplot(
                                x=subset["PlateLocSide"],
                                y=subset["PlateLocHeight"],
                                fill=True,
                                levels=10,
                                thresh=0.05,
                                bw_adjust=0.8,
                                cmap="Reds",
                                ax=ax,
                                alpha=0.7
                            )
                        except:
                            ax.text(0.5, 0.5, "Low Density", ha="center", va="center", transform=ax.transAxes)

                    ax.set_xlim(-2.0, 2.0)
                    ax.set_ylim(0, 5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"{family}\n(n={len(subset)})", fontsize=14, fontweight='bold')
                    
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                    st.pyplot(fig)
                    plt.close(fig)

elif app == "NCAA Pitcher":
    # --- HEADER SECTION ---
    st.subheader("⚾ NCAA Pitcher Analytics")

    # --- SIDEBAR FILTERS ---
    # 1. Team Selection
    teams = load_team_names("pitcher")
    sel_team = st.sidebar.selectbox("Select Team", teams, index=0, key="ncaa_p_team")
    
    # 2. Pitcher Selection (Dependent on Team)
    pitchers = load_players_for_team(sel_team, "pitcher")
    
    # Format names: "O'Connor, Andrew" -> "Andrew O'Connor"
    pitchers_fmt = [' '.join(p.split(', ')[::-1]) if ', ' in p else p for p in pitchers]
    sel_pitcher_fmt = st.sidebar.selectbox("Select Pitcher", pitchers_fmt, key="ncaa_p_player")
    
    # Convert back to raw format for data loading
    sel_pitcher_raw = ', '.join(sel_pitcher_fmt.split(' ')[::-1]) if ' ' in sel_pitcher_fmt else sel_pitcher_fmt

    # 3. Load Data (ONCE for all tabs)
    data = load_player_data(sel_pitcher_raw, sel_team, "pitcher")

    # --- BIO HEADER ---
    if not data.empty:
        p_info = data.iloc[0]
        p_height = p_info['Height_Pitcher'] if pd.notnull(p_info['Height_Pitcher']) else "N/A"
        p_weight = f"{int(p_info['Weight_Pitcher'])} lbs" if pd.notnull(p_info['Weight_Pitcher']) else "N/A"
        p_jersey = f"#{int(p_info['Jersey_Pitcher'])}" if pd.notnull(p_info['Jersey_Pitcher']) else ""
        
        st.header(f"{sel_pitcher_fmt} {p_jersey}")
        st.subheader(f"{sel_team} • {p_info['PitcherThrows']}HP")
        st.write(f"**Physicals:** {p_height} | {p_weight}")
        st.divider()

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Performance Data", "Stuff Visuals", "Sequencing", "Usage", "Heatmaps", "Trends"])

    # --- TAB 1: PERFORMANCE DATA ---
    with tab1:
        if not data.empty:
            # Clean data
            df_tab1 = data[~data['TaggedPitchType'].isin(['Undefined', 'Other'])].copy()
            
            # Force numeric
            for col in ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'Extension']:
                df_tab1[col] = pd.to_numeric(df_tab1[col], errors='coerce')

            # Summary Table
            summary = df_tab1.groupby('TaggedPitchType').agg(
                Count=('TaggedPitchType', 'count'),
                AvgVelo=('RelSpeed', 'mean'),
                MaxVelo=('RelSpeed', 'max'),
                AvgSpin=('SpinRate', 'mean'),
                AvgIVB=('InducedVertBreak', 'mean'),
                AvgHB=('HorzBreak', 'mean'),
                AvgExt=('Extension', 'mean')
            ).reset_index()
            
            total_p = summary['Count'].sum()
            summary['Usage'] = (summary['Count'] / total_p * 100).round(1)
            summary = summary.rename(columns={'TaggedPitchType': 'Pitch', 'Count': '#'})
            
            st.subheader("Pitch Shape Summary")
            st.dataframe(
                summary[['Pitch', '#', 'Usage', 'AvgVelo', 'MaxVelo', 'AvgSpin', 'AvgIVB', 'AvgHB', 'AvgExt']].style.format({
                    'Usage': '{:.1f}%', 'AvgVelo': '{:.1f}', 'MaxVelo': '{:.1f}',
                    'AvgSpin': '{:.0f}', 'AvgIVB': '{:.1f}', 'AvgHB': '{:.1f}', 'AvgExt': '{:.2f}'
                }), use_container_width=True
            )

            # Splits
            for side in ['Left', 'Right']:
                st.subheader(f"vs {side}-Handed Hitters")
                split_df = df_tab1[df_tab1['BatterSide'] == side]
                if not split_df.empty:
                    stats_df = pitch_type_stats(split_df)
                    if not stats_df.empty:
                        overall_row = pd.DataFrame([overall_stats(split_df)])
                        final_df = pd.concat([overall_row, stats_df], ignore_index=True)
                        st.dataframe(final_df.style.format({
                            'Usage': '{:.1f}%', 'AVG': '{:.3f}', 'SLG': '{:.3f}',
                            'Zone%': '{:.1f}%', 'Whiff%': '{:.1f}%', 'Zone Whiff%': '{:.1f}%',
                            'Chase%': '{:.1f}%', 'run_value': '{:.2f}', 'wOBA': '{:.3f}',
                            'xwOBA': '{:.3f}', 'HH%': '{:.1f}%', 'GB%': '{:.1f}%'
                        }), use_container_width=True)
                else:
                    st.info(f"No pitch data found against {side}-handed hitters.")
        else:
            st.info("Select a pitcher to view data.")

    # --- TAB 2: STUFF VISUALS ---
    with tab2:
        # Filters (Date Only)
        c1, c2 = st.columns([1, 3]) 
        with c1:
            date_range = st.date_input("Date Range", [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-12-31")], key="ncaa_p_date_tab2")

        if not data.empty:
            filtered_data = data.copy()
            filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
            
            # Numeric conversion for charts
            for col in ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'PlateLocSide', 'PlateLocHeight']:
                if col in filtered_data.columns:
                    filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

            # Apply Date Filter
            if len(date_range) == 2:
                filtered_data = filtered_data[(filtered_data['Date'] >= pd.to_datetime(date_range[0])) & 
                                              (filtered_data['Date'] <= pd.to_datetime(date_range[1]))]

            if filtered_data.empty:
                st.warning("No data found for this date range.")
            else:
                # --- LOCAL FIXING WIDGET (Admin) ---
                def show_admin_fix_widget(selected_data, chart_name):
                    if not st.session_state.get("is_admin", False): return
                    if selected_data and "selection" in selected_data:
                        pts = selected_data["selection"]["points"]
                        if not pts: return
                        
                        safe_indices = [int(p["customdata"][0]) for p in pts if "customdata" in p]
                        
                        if safe_indices:
                            st.info(f"🔍 Selected {len(safe_indices)} pitches.")
                            c1, c2 = st.columns([2, 1])
                            with c1:
                                new_tag = st.selectbox("Change to:", list(PITCH_PALETTE.keys()), key=f"fix_{chart_name}")
                            with c2:
                                st.write("") 
                                if st.button(f"✅ Apply Fix", key=f"btn_{chart_name}"):
                                    try:
                                        path = get_local_data_path()
                                        full_df = pd.read_parquet(path)
                                        full_df.loc[safe_indices, 'TaggedPitchType'] = new_tag
                                        full_df.to_parquet("ncaa_data_2025_fixed.parquet", index=False)
                                        st.cache_data.clear()
                                        st.success("✅ Fixed! Download below.")
                                    except Exception as e: st.error(f"Error: {e}")

                # 1. MOVEMENT PROFILE
                st.subheader("Interactive Movement Profile (IVB vs HB)")
                st.caption("Lasso to inspect or fix pitch tags.")
                
                fig_mov = px.scatter(
                    filtered_data, x="HorzBreak", y="InducedVertBreak", 
                    color="TaggedPitchType", color_discrete_map=PITCH_PALETTE,
                    hover_data=["RelSpeed", "SpinRate", "Date"], width=650, height=650
                )
                fig_mov.add_hline(y=0, line_dash="dash", line_color="black")
                fig_mov.add_vline(x=0, line_dash="dash", line_color="black")
                fig_mov.update_layout(
                    xaxis=dict(range=[-30, 30], title="Horizontal Break (in)"), 
                    yaxis=dict(range=[-30, 30], title="Induced Vertical Break (in)", scaleanchor="x", scaleratio=1),
                    plot_bgcolor='white', dragmode='lasso'
                )
                fig_mov.update_traces(customdata=filtered_data.index) 

                selection_mov = st.plotly_chart(fig_mov, on_select="rerun", selection_mode=["box", "lasso"], key="ncaa_mov_scatter")
                show_admin_fix_widget(selection_mov, "ncaa_movement_chart")
                
                st.divider()

                # 2. VELOCITY vs SPIN RATE
                st.subheader("Velocity vs. Spin Rate")
                st.caption("Identify misclassified pitches by spin/velo clusters.")

                fig_ss = px.scatter(
                    filtered_data, x='RelSpeed', y='SpinRate', 
                    color='TaggedPitchType', color_discrete_map=PITCH_PALETTE,
                    labels={'RelSpeed': 'Velocity (MPH)', 'SpinRate': 'Spin Rate (RPM)'},
                    hover_data=['RelSpeed', 'SpinRate', 'Date'], width=750, height=500
                )
                fig_ss.update_layout(plot_bgcolor='white', dragmode='lasso')
                fig_ss.update_traces(customdata=filtered_data.index, marker=dict(size=8, line=dict(width=1, color='white')))

                selection_ss = st.plotly_chart(fig_ss, on_select="rerun", selection_mode=["box", "lasso"], key="ncaa_velo_spin_scatter")
                show_admin_fix_widget(selection_ss, "ncaa_velo_spin_chart")

                st.divider()

                # 3. PERFORMANCE BY ZONE
                st.subheader("📍 Performance by Zone")
                z_col1, z_col2 = st.columns(2)
                with z_col1:
                    st.plotly_chart(draw_performance_grid(filtered_data, 'Left'), use_container_width=True)
                with z_col2:
                    st.plotly_chart(draw_performance_grid(filtered_data, 'Right'), use_container_width=True)

                # 4. ADMIN DOWNLOAD
                if st.session_state.get("is_admin", False):
                    st.divider()
                    st.markdown("### 💾 Save Your Work")
                    if os.path.exists("ncaa_data_2025_fixed.parquet"):
                        with open("ncaa_data_2025_fixed.parquet", "rb") as f:
                            st.download_button("⬇️ Download Fixed Data", f, "ncaa_data_2025.parquet", "application/octet-stream")
                    else:
                        st.info("Make a change above to generate a downloadable file.")
        else:
            st.info("Select a pitcher in the sidebar to view data.")



    with tab3:
        st.subheader(f"Pitch Sequencing")

        def pitch_sequencing_section(seq_data, label):
            st.markdown(f"### {label}")

            # Prepare data sorted by PA and pitch number
            seq_data = seq_data.sort_values(['Batter', 'Pitcher', 'PAofInning', 'PitchofPA'])

            # Identify previous pitch type for each pitch within a PA
            seq_data['PrevPitchType'] = seq_data.groupby(['Batter', 'Pitcher', 'PAofInning'])['TaggedPitchType'].shift(1)
            seq_data['Sequence'] = seq_data['PrevPitchType'] + '/' + seq_data['TaggedPitchType']

            # Filter out first pitches (no previous pitch)
            seq_data = seq_data[seq_data['PrevPitchType'].notnull()]

            total_sequences = len(seq_data)

            # Most common sequences (as percentage)
            common_seqs = seq_data['Sequence'].value_counts(normalize=True).reset_index()
            common_seqs.columns = ['Sequence', 'Percentage']
            common_seqs['Percentage'] = (common_seqs['Percentage'] * 100).round(1)
            st.write("**Most Common Sequences (%):**")
            st.dataframe(common_seqs.head(10))

            # Sequence outcomes
            def sequence_outcome_table(df, mask, label):
                filtered = df[mask]
                total = len(filtered)
                outcome_seqs = filtered['Sequence'].value_counts(normalize=True).reset_index()
                outcome_seqs.columns = ['Sequence', 'Percentage']
                outcome_seqs['Percentage'] = (outcome_seqs['Percentage'] * 100).round(1)
                st.write(f"**Sequences Leading to {label} (%):**")
                st.dataframe(outcome_seqs.head(10))

            # Whiffs: PitchCall == 'StrikeSwinging'
            whiff_mask = seq_data['PitchCall'] == 'StrikeSwinging'
            sequence_outcome_table(seq_data, whiff_mask, 'Whiffs')

            # Weak contact: ExitSpeed < 80 (only for pitches in play)
            weak_mask = (seq_data['PitchCall'] == 'InPlay') & (pd.to_numeric(seq_data['ExitSpeed'], errors='coerce') < 80)
            sequence_outcome_table(seq_data, weak_mask, 'Weak Contact (ExitSpeed < 80)')

            # Damage: ExitSpeed > 95 (only for pitches in play)
            damage_mask = (seq_data['PitchCall'] == 'InPlay') & (pd.to_numeric(seq_data['ExitSpeed'], errors='coerce') > 95)
            sequence_outcome_table(seq_data, damage_mask, 'Damage (ExitSpeed > 95)')

            # Count-specific sequences (e.g., 0-2, 1-2, 3-2) as percentage
            st.write("**Count-Specific Sequences (% for 0-2, 1-2, 3-2):**")
            seq_data['Count'] = seq_data['Balls'].astype(str) + '-' + seq_data['Strikes'].astype(str)
            for count in ['0-2', '1-2', '3-2']:
                count_seqs = seq_data[seq_data['Count'] == count]['Sequence'].value_counts(normalize=True).reset_index()
                count_seqs.columns = ['Sequence', 'Percentage']
                count_seqs['Percentage'] = (count_seqs['Percentage'] * 100).round(1)
                st.write(f"Most common sequences for count {count}:")
                st.dataframe(count_seqs.head(5))

        # Split by vLHH and vRHH
        vLHH_seq_data = data[data['BatterSide'] == 'Left']
        vRHH_seq_data = data[data['BatterSide'] == 'Right']

        pitch_sequencing_section(vLHH_seq_data, "vs Left-Handed Hitters (vLHH)")
        pitch_sequencing_section(vRHH_seq_data, "vs Right-Handed Hitters (vRHH)")

    with tab4:
        st.subheader(f"Pitch Usage Pie Charts")

        import matplotlib.pyplot as plt

        # Split data by vLHH and vRHH
        vLHH_data = data[data['BatterSide'] == 'Left']
        vRHH_data = data[data['BatterSide'] == 'Right']

        pitch_palette = {
            'Fastball': '#D22D49',  # 4-Seam Red
            'Four-Seam': '#D22D49',
            'Sinker': '#FE9D00',    # Statcast Orange
            'One-Seam Fastball': '#FE9D00',
            'Two-Seam Fastball': '#FE9D00',
            'Cutter': '#933F2C',    # Statcast Maroon
            'Slider': '#DDB33A',    # Statcast Yellow
            'Sweeper': '#DDB33A',   # Statcast Gold
            'Curveball': '#00D1ED', # Statcast Cyan
            'Knuckle Curve': '#6236CD', # Statcast Purple
            'Changeup': '#1DBE3A',  # Statcast Green
            'Splitter': '#3BACAC',  # Statcast Teal
            'Knuckleball': '#867A08',
            'Other': '#999999'
        }

        # ---- Add legend at the top, only for used pitch types ----
        used_types = set(vLHH_data['TaggedPitchType'].unique()) | set(vRHH_data['TaggedPitchType'].unique())
        legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px;'>"
        for pitch, color in pitch_palette.items():
            if pitch in used_types:
                legend_html += f"<div style='display: flex; align-items: center; gap: 6px;'><div style='width: 16px; height: 16px; background: {color}; border-radius: 4px; border: 1px solid #888;'></div><span style='font-size: 14px;'>{pitch}</span></div>"
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

        def plot_pie_chart(df, mask_type):
            df = df.copy()
            df['Balls'] = pd.to_numeric(df['Balls'], errors='coerce').fillna(-1).astype(int)
            df['Strikes'] = pd.to_numeric(df['Strikes'], errors='coerce').fillna(-1).astype(int)
            if mask_type == "first":
                mask = (df['Balls'] == 0) & (df['Strikes'] == 0)
            elif mask_type == "hitter":
                mask = (df['Balls'] > df['Strikes'])
            elif mask_type == "pitcher":
                mask = (df['Strikes'] > df['Balls'])
            else:
                mask = pd.Series([True] * len(df))
            usage = df.loc[mask, 'TaggedPitchType'].value_counts()
            fig, ax = plt.subplots(figsize=(3, 3))  # All pies same size
            if not usage.empty:
                colors = [pitch_palette.get(pt, 'grey') for pt in usage.index]
                wedges, texts, autotexts = ax.pie(
                    usage,
                    labels=None,  # No labels on slices
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=140,
                    textprops={'fontsize': 12}
                )
                ax.axis('equal')  # Always a perfect circle
                for autotext in autotexts:
                    autotext.set_fontsize(12)
                ax.set_title("", fontsize=12)
                plt.tight_layout()
                return fig
            else:
                plt.close(fig)
                return None

        chart_types = [
            ("first", "First Pitch"),
            ("hitter", "Hitter's Count"),
            ("pitcher", "Pitcher's Ahead")
        ]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### vLHH")
        with col2:
            st.markdown("### vRHH")

        for mask_type, chart_title in chart_types:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{chart_title}**")
                fig = plot_pie_chart(vLHH_data, mask_type)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info(f"No pitch data available for vLHH {chart_title}.")
            with col2:
                st.markdown(f"**{chart_title}**")
                fig = plot_pie_chart(vRHH_data, mask_type)
                if fig:
                    st.pyplot(fig)
                else:
                   st.info(f"No pitch data available for vRHH {chart_title}.")

    with tab5:
        st.header("Strike Zone Heatmaps")
        st.caption("Visualizing pitch density and location strategy.")

        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt
        import seaborn as sns

        # --- 1. DATA PREPARATION (Fixing the Numeric Error) ---
        df_heat = data.copy()

        # Force critical columns to numeric to prevent "categorical" errors
        plot_cols = ["PlateLocSide", "PlateLocHeight", "ExitSpeed"]
        for col in plot_cols:
            df_heat[col] = pd.to_numeric(df_heat[col], errors='coerce')

        # Drop rows without location data (KDE plots will fail without coordinates)
        df_heat = df_heat.dropna(subset=["PlateLocSide", "PlateLocHeight"])

        # --- 2. HEATMAP FILTERS ---
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            map_types = ["All Pitches", "Whiffs", "Hard Hit (95+)", "Softly Hit (<80)", "Chases", "Called Strikes"]
            map_sel = st.selectbox("Select Metric", map_types, key="hmap_metric_final")
        with h_col2:
            side_sel = st.radio("Batter Side", ["Combined", "Left", "Right"], horizontal=True, key="hmap_side_final")

        # Filter by Batter Side
        if side_sel != "Combined":
            df_heat = df_heat[df_heat["BatterSide"] == side_sel]

        # Filter by Event Type
        if map_sel == "All Pitches":
            df_event = df_heat
        elif map_sel == "Whiffs":
            df_event = df_heat[df_heat["PitchCall"] == "StrikeSwinging"]
        elif map_sel == "Hard Hit (95+)":
            df_event = df_heat[df_heat["ExitSpeed"] >= 95]
        elif map_sel == "Softly Hit (<80)":
            df_event = df_heat[df_heat["ExitSpeed"] <= 80]
        elif map_sel == "Chases":
            swing_calls = ["strikeswinging", "foul", "inplay"]
            # Strike Zone is roughly -0.83 to 0.83 horizontally, 1.5 to 3.5 vertically
            in_zone = df_heat["PlateLocSide"].between(-0.83, 0.83) & df_heat["PlateLocHeight"].between(1.5, 3.5)
            df_event = df_heat[df_heat["PitchCall"].str.lower().isin(swing_calls) & ~in_zone]
        else: # Called Strikes
            df_event = df_heat[df_heat["PitchCall"] == "StrikeCalled"]

        # --- 3. RENDER HEATMAPS ---
        if df_event.empty:
            st.warning(f"No data points found for {map_sel} vs {side_sel} hitters.")
        else:
            # Get top 5 pitch types by frequency
            top5_pitches = df_event["TaggedPitchType"].value_counts().index.tolist()[:5]
            
            # Create columns based on how many pitch types exist
            h_cols = st.columns(len(top5_pitches))
            
            for i, col in enumerate(h_cols):
                with col:
                    pt_type = top5_pitches[i]
                    subset = df_event[df_event["TaggedPitchType"] == pt_type]

                    # Create the Matplotlib Figure
                    fig, ax = plt.subplots(figsize=(4, 5))
                    
                    # Draw Strike Zone (Black Outline)
                    ax.add_patch(Rectangle((-0.83, 1.5), 1.66, 2.0, 
                                        fill=False, edgecolor="black", linewidth=2.5, zorder=10))
                    
                    # Draw Home Plate at the bottom for orientation
                    ax.plot([-0.83, 0.83, 0.83, 0, -0.83, -0.83], [0, 0, 0.15, 0.3, 0.15, 0], color="black", lw=1.5)

                    if len(subset) < 5:
                        ax.text(0.5, 0.5, "Not Enough\nData", ha="center", va="center", transform=ax.transAxes, fontsize=12)
                    else:
                        # Create Density Plot
                        sns.kdeplot(
                            x=subset["PlateLocSide"],
                            y=subset["PlateLocHeight"],
                            fill=True,
                            levels=10,
                            thresh=0.02,
                            bw_adjust=0.7,
                            cmap="Reds",
                            ax=ax,
                            alpha=0.7
                        )

                    # Visual Settings
                    ax.set_xlim(-2.5, 2.5) # Wide enough to see "Chase" pitches
                    ax.set_ylim(0, 5)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f"{pt_type}\n(n={len(subset)})", fontsize=14, fontweight='bold')
                    
                    st.pyplot(fig)
                    plt.close(fig) # Memory management

    with tab6:
        st.subheader("📈 Season Trends")
        st.caption("Track changes in velocity, spin, and movement profile game-by-game.")
        if not data.empty:
            plot_trend_lines(data)
        else:
            st.warning("No data available.")




elif app == "Holy Cross Hitter":
    st.subheader("✝️ Holy Cross Hitter Stats")

    # Hard-coded Team Selection
    selected_team_full = "College of the Holy Cross"
    
    # Step 1: Load ONLY player names for Holy Cross
    batters = load_players_for_team(selected_team_full, app_type="hitter")
    batters_formatted = [' '.join(b.split(', ')[::-1]) if ', ' in b else b for b in batters]
    selected_batter_fmt = st.sidebar.selectbox("Select Holy Cross Batter", options=batters_formatted)
    
    # Convert back to raw format
    selected_batter_raw = ', '.join(selected_batter_fmt.split(' ')[::-1]) if ' ' in selected_batter_fmt else selected_batter_fmt

    # Step 2: Load ONLY this player's data (lazy loaded!)
    player_data = load_player_data(selected_batter_raw, selected_team_full, app_type="hitter")
    
    # --- PLAYER BIO HEADER ---
    if not player_data.empty:
        p_info = player_data.iloc[0]
        p_height = p_info['Height_Batter'] if pd.notnull(p_info['Height_Batter']) else "N/A"
        p_weight = f"{int(p_info['Weight_Batter'])} lbs" if pd.notnull(p_info['Weight_Batter']) else ""
        p_jersey = f"#{int(p_info['Jersey_Batter'])}" if pd.notnull(p_info['Jersey_Batter']) else ""
        
        batter_side = p_info['BatterSide']
        hand_display = "LHH" if str(batter_side).upper().startswith("L") else "RHH" if str(batter_side).upper().startswith("R") else ""

        st.header(f"{selected_batter_fmt} {p_jersey}")
        st.subheader(f"{selected_team_full} | {hand_display}")
        st.write(f"**Physical Profile:** {p_height} | {p_weight}")
        st.divider()

        # --- CALCS ---
        overall = get_advanced_metrics(player_data)
        lhp_data = player_data[player_data["PitcherThrows"].str.contains('L|l', na=False)]
        rhp_data = player_data[player_data["PitcherThrows"].str.contains('R|r', na=False)]
        l_m = get_advanced_metrics(lhp_data)
        r_m = get_advanced_metrics(rhp_data)
        
        tab1, tab2, tab3 = st.tabs(["Stats Overview", "Spray Chart", "Heat Maps"])

        with tab1:
            # --- 1. Overall Production Table ---
            st.markdown("#### Overall Production")
            overall_df = pd.DataFrame({
                "AVG": [f"{overall['AVG']:.3f}"], 
                "OBP": [f"{overall['OBP']:.3f}"], 
                "SLG": [f"{overall['SLG']:.3f}"], 
                "wOBA": [f"{overall['wOBA']:.3f}"], 
                "xwOBA": [f"{overall['xwOBA']:.3f}"]
            })
            st.table(overall_df)
            
            # --- 2. Contact & Plate Discipline Table ---
            st.markdown("#### Contact & Plate Discipline")
            discipline_df = pd.DataFrame({
                "90th EV": [f"{overall['90th EV']:.1f}"],
                "Hard Hit%": [f"{overall['HardHit%']:.1f}%"],
                "BB%": [f"{overall['BB%']:.1f}%"],
                "K%": [f"{overall['K%']:.1f}%"],
                "Chase%": [f"{overall['Chase%']:.1f}%"],
                "Z-Contact%": [f"{overall['Z-Contact%']:.1f}%"],
                "LD%": [f"{overall['LD%']:.1f}%"],
                "GB%": [f"{overall['GB%']:.1f}%"],
                "FB%": [f"{overall['FB%']:.1f}%"],
                "HHLD%": [f"{overall['HHLD%']:.1f}%"]
            })
            st.table(discipline_df)

            st.divider()

            # --- 3. Split Performance Logic ---
            def get_pitch_group_splits(df):
                """Categorizes pitches and calculates metrics for each group."""
                fastballs = ['Fastball', 'Sinker', 'FourSeamFastBall', 'TwoSeamFastBall', 'Cutter']
                breaking = ['Slider', 'Curveball', 'Sweeper', 'KnuckleCurve']
                offspeed = ['ChangeUp', 'Splitter', 'Knuckleball']
                
                df = df.copy()
                df['Category'] = 'Other'
                df.loc[df['TaggedPitchType'].isin(fastballs), 'Category'] = 'Fastballs'
                df.loc[df['TaggedPitchType'].isin(breaking), 'Category'] = 'Breaking'
                df.loc[df['TaggedPitchType'].isin(offspeed), 'Category'] = 'Off-Speed'
                
                rows = []
                rows.append({"Type": "OVERALL", **get_advanced_metrics(df)})
                for cat in ['Fastballs', 'Breaking', 'Off-Speed']:
                    cat_df = df[df['Category'] == cat]
                    if not cat_df.empty:
                        rows.append({"Type": cat.upper(), **get_advanced_metrics(cat_df)})
                
                res_df = pd.DataFrame(rows)
                formatted = pd.DataFrame({
                    "Pitch Type": res_df['Type'],
                    "Pitches": res_df['Pitches'],
                    "Avg EV": res_df['Avg EV'].map("{:.1f}".format),
                    "Avg LA": res_df['Avg LA'].map("{:.1f}°".format),
                    "OPS": res_df['OPS'].map("{:.3f}".format),
                    "wOBA": res_df['wOBA'].map("{:.3f}".format),
                    "HardHit%": res_df['HardHit%'].map("{:.1f}%".format),
                    "HHLD%": res_df['HHLD%'].map("{:.1f}%".format),
                    "Contact%": res_df['Contact%'].map("{:.1f}%".format),
                    "Whiff%": res_df['Miss%'].map("{:.1f}%".format),
                    "Chase%": res_df['Chase%'].map("{:.1f}%".format),
                    "K%": res_df['K%'].map("{:.1f}%".format)
                })
                return formatted

            # --- vs LHP Performance ---
            st.markdown("#### vs Left-Handed Pitchers")
            if not lhp_data.empty:
                lhp_split_df = get_pitch_group_splits(lhp_data)
                st.dataframe(lhp_split_df.style.hide(axis='index'), use_container_width=True)
            else:
                st.info("No data available vs Left-Handed Pitchers")

            # --- vs RHP Performance ---
            st.markdown("#### vs Right-Handed Pitchers")
            if not rhp_data.empty:
                rhp_split_df = get_pitch_group_splits(rhp_data)
                st.dataframe(rhp_split_df.style.hide(axis='index'), use_container_width=True)
            else:
                st.info("No data available vs Right-Handed Pitchers")

        with tab2:
            st.subheader(f"Spray Chart Analysis: {selected_batter_fmt}")
            
            # 1. Prepare and Clean Data
            spray_df = player_data.copy()
            spray_df['Distance'] = pd.to_numeric(spray_df['Distance'], errors='coerce')
            spray_df['Direction'] = pd.to_numeric(spray_df['Direction'], errors='coerce')
            spray_df['ExitSpeed'] = pd.to_numeric(spray_df['ExitSpeed'], errors='coerce')
            
            # Filter: Exclude undefined and foul results
            to_exclude = ['undefined', 'foul', 'null', 'nan']
            spray_df = spray_df[~spray_df['PlayResult'].astype(str).str.lower().isin(to_exclude)]
            spray_df = spray_df.dropna(subset=['Distance', 'Direction'])

            if not spray_df.empty:
                # Create two columns layout
                col_spray, col_heat = st.columns([1.5, 1])
                
                with col_spray:
                    # 2. Coordinate Calculation
                    spray_df['hc_x'] = spray_df['Distance'] * np.sin(np.deg2rad(spray_df['Direction']))
                    spray_df['hc_y'] = spray_df['Distance'] * np.cos(np.deg2rad(spray_df['Direction']))
                    
                    fig = go.Figure()

                    # Custom Color Palette
                    color_map = {
                        'single': '#FFD700', 'double': '#00CD66', 'triple': '#00F5FF', 
                        'homerun': '#9370DB', 'home run': '#9370DB',
                        'sacrifice': '#708090', 'fielderschoice': '#708090', 
                        'error': '#708090', 'out': '#708090'
                    }
                    
                    # 3. Add Data Traces
                    for res, group in spray_df.groupby('PlayResult'):
                        res_key = res.lower().replace(" ", "").strip()
                        fig.add_trace(go.Scatter(
                            x=group['hc_x'], y=group['hc_y'], mode='markers', name=res.title(),
                            customdata=np.stack((group['PlayResult'], group['ExitSpeed'], group['Distance']), axis=-1),
                            marker=dict(
                                size=10, 
                                color=color_map.get(res_key, '#708090'), 
                                opacity=0.8, 
                                line=dict(width=1, color='white')
                            ),
                            hovertemplate="<b>Result:</b> %{customdata[0]}<br><b>EV:</b> %{customdata[1]:.1f} mph<br><b>Dist:</b> %{customdata[2]:.0f} ft<extra></extra>"
                        ))
                    
                    # 4. Field Layout
                    fig.update_layout(
                        shapes=draw_baseball_field(),
                        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False, range=[-20, 450]),
                        xaxis=dict(visible=False, range=[-300, 300]),
                        width=700, height=650, plot_bgcolor='white',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_heat:
                    st.markdown("#### 🎯 Hit Tendency")
                    # CALL THE HELPER FUNCTION HERE
                    draw_tendency_heatmap(spray_df)
                    
                    # Quick Text Summary
                    pull_ev = spray_df[spray_df['Direction'] < -15]['ExitSpeed'].mean()
                    oppo_ev = spray_df[spray_df['Direction'] > 15]['ExitSpeed'].mean()
                    st.metric("Avg EV (Pull)", f"{pull_ev:.1f} mph")
                    st.metric("Avg EV (Oppo)", f"{oppo_ev:.1f} mph")

            else:
                st.warning("No valid fair-ball data available.")

        with tab3:
            st.subheader("Launch Angle vs. Exit Velocity (Airborne Fair Balls)")
            
            # --- 1. DATA PREP FOR SCATTER PLOT ---
            plot_df = player_data.copy()
            plot_df['ExitSpeed'] = pd.to_numeric(plot_df['ExitSpeed'], errors='coerce')
            plot_df['Angle'] = pd.to_numeric(plot_df['Angle'], errors='coerce')
            
            plot_df['PlayResult_Lower'] = plot_df['PlayResult'].fillna('').astype(str).str.lower().str.strip()
            
            to_exclude = ['undefined', 'foul', 'null', 'nan', '']
            plot_df = plot_df[
                (plot_df['Angle'] >= 0) & 
                (~plot_df['PlayResult_Lower'].isin(to_exclude))
            ].copy()
            
            plot_df = plot_df.dropna(subset=['ExitSpeed', 'Angle'])

            # --- 2. SCATTER PLOT & PERFORMANCE GRID ---
            if not plot_df.empty:
                fig = px.scatter(
                    plot_df, x='ExitSpeed', y='Angle', color='PlayResult_Lower',
                    labels={'ExitSpeed': 'Exit Velocity (mph)', 'Angle': 'Launch Angle (deg)', 'PlayResult_Lower': 'Result'},
                    color_discrete_map={
                        'single': '#FFD700', 'double': '#00CD66', 'triple': '#00F5FF', 
                        'homerun': '#9370DB', 'home run': '#9370DB', 'sacrifice': '#708090', 
                        'fielderschoice': '#708090', 'error': '#708090', 'out': '#708090'
                    },
                    hover_data={'ExitSpeed': ':.1f', 'Angle': ':.1f', 'PlayResult': True}
                )
                fig.add_hrect(y0=10, y1=30, line_width=0, fillcolor="red", opacity=0.1, 
                              annotation_text="Power Alley (10°-30°)", annotation_position="top left")
                fig.add_vline(x=95, line_dash="dash", line_color="black", annotation_text="Hard Hit (95+)")
                fig.update_layout(width=800, height=600, plot_bgcolor='white', legend_title_text='Play Result')
                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                st.subheader("📊 Performance by Zone")
                
                def draw_performance_grid(data, hand):
                    hand_col = 'PitcherThrows' if 'PitcherThrows' in data.columns else 'PitcherHand'
                    
                    df = data[data[hand_col] == hand].copy()
                    df['PlateLocSide'] = pd.to_numeric(df['PlateLocSide'], errors='coerce')
                    df['PlateLocHeight'] = pd.to_numeric(df['PlateLocHeight'], errors='coerce')
                    
                    # Define Zones
                    df['Zone'] = 'Outside'
                    df.loc[(df['PlateLocSide'] < 0) & (df['PlateLocHeight'] > 2.5), 'Zone'] = 'Upper Left'
                    df.loc[(df['PlateLocSide'] >= 0) & (df['PlateLocHeight'] > 2.5), 'Zone'] = 'Upper Right'
                    df.loc[(df['PlateLocSide'] < 0) & (df['PlateLocHeight'] <= 2.5), 'Zone'] = 'Lower Left'
                    df.loc[(df['PlateLocSide'] >= 0) & (df['PlateLocHeight'] <= 2.5), 'Zone'] = 'Lower Right'

                    fig_zone = go.Figure()
                    
                    # Coordinates for the 4 zones [x0, y0, x1, y1]
                    quads = [[[-0.83, 2.5, 0, 3.5], 'Upper Left'], [[0, 2.5, 0.83, 3.5], 'Upper Right'],
                            [[-0.83, 1.5, 0, 2.5], 'Lower Left'], [[0, 1.5, 0.83, 2.5], 'Lower Right']]

                    for coords, name in quads:
                        z_df = df[df['Zone'] == name]
                        if not z_df.empty:
                            m = get_advanced_metrics(z_df)
                            
                            # COLOR LOGIC: Darker Red/Green so White text is readable
                            # Green = Forest Green (High Opacity)
                            # Red = Firebrick (High Opacity)
                            bg_color = "rgba(34, 139, 34, 0.9)" if m['OPS'] >= 0.800 else "rgba(178, 34, 34, 0.9)"
                            
                            # 1. Draw the Colored Rectangle (Layer=Below ensures text sits on top)
                            fig_zone.add_shape(type="rect", 
                                x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3],
                                line=dict(color="white", width=1), 
                                fillcolor=bg_color,
                                layer="below" 
                            )
                            
                            # 2. Draw the Text (Bright White & Bold)
                            fig_zone.add_trace(go.Scatter(
                                x=[(coords[0]+coords[2])/2], y=[(coords[1]+coords[3])/2],
                                text=f"OPS: {m['OPS']:.3f}<br>AVG: {m['AVG']:.3f}<br>EV: {m['Avg EV']:.1f}",
                                mode="text", 
                                textfont=dict(size=14, color="white", weight="bold"), # <--- THE KEY FIX
                                showlegend=False
                            ))
                        else:
                            # Empty Zone (Gray)
                            fig_zone.add_shape(type="rect", 
                                x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3],
                                line=dict(color="gray", width=1), 
                                fillcolor="rgba(100, 100, 100, 0.3)",
                                layer="below"
                            )

                    # Draw Home Plate
                    fig_zone.add_shape(type="path", path="M -0.4 0 L 0.4 0 L 0.4 0.2 L 0 0.4 L -0.4 0.2 Z",
                                line=dict(color="white", width=2), fillcolor="gray", layer="below")

                    fig_zone.update_layout(
                        title=dict(text=f"vs {hand}HP", x=0.5, font=dict(size=16, color="white")),
                        xaxis=dict(range=[-1.5, 1.5], visible=False),
                        yaxis=dict(range=[-0.5, 4.0], visible=False), 
                        width=350, height=450,
                        margin=dict(l=10, r=10, t=40, b=10), 
                        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    return fig_zone

                z_col1, z_col2 = st.columns(2)
                with z_col1:
                    st.plotly_chart(draw_performance_grid(player_data, 'Left'), use_container_width=True)
                with z_col2:
                    st.plotly_chart(draw_performance_grid(player_data, 'Right'), use_container_width=True)

            else:
                st.warning("No airborne contact data available.")

            st.divider()

            # --- 3. ADVANCED HEATMAPS (Runs independently of Scatter Plot) ---
            st.subheader("🔥 Zone Heatmaps")
            st.caption("Pitch density visualization by pitch family.")

            # Imports
            from matplotlib.patches import Rectangle
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Prepare Data
            df_heat = player_data.copy()

            # Force numeric types
            for col in ["PlateLocSide", "PlateLocHeight", "ExitSpeed"]:
                df_heat[col] = pd.to_numeric(df_heat[col], errors='coerce')

            # Drop invalid rows
            df_heat = df_heat.dropna(subset=["PlateLocSide", "PlateLocHeight"])

            # Define Pitch Families
            def get_pitch_family(pitch_type):
                pitch_type = str(pitch_type).lower()
                if any(x in pitch_type for x in ['fastball', 'sinker', 'cutter', 'four', 'seam']): return 'Fastballs'
                if any(x in pitch_type for x in ['slider', 'curve', 'sweeper', 'slurve', 'knucklecurve']): return 'Breaking'
                if any(x in pitch_type for x in ['change', 'split', 'knuckle', 'fork']): return 'Off-Speed'
                return 'Other'

            df_heat['PitchFamily'] = df_heat['TaggedPitchType'].apply(get_pitch_family)

            # Filters
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                map_types = ["All Pitches", "Whiffs", "Hard Hit (95+)", "Softly Hit (<80)", "Chases", "Called Strikes"]
                map_sel = st.selectbox("Select Metric", map_types, key="hmap_metric_hitter_tab3")
            
            with h_col2:
                pitcher_side_sel = st.radio("Pitcher Throws", ["Combined", "Right", "Left"], horizontal=True, key="hmap_side_hitter_tab3")

            # Apply Pitcher Hand Filter
            if pitcher_side_sel == "Right":
                df_heat = df_heat[df_heat["PitcherThrows"] == "Right"]
            elif pitcher_side_sel == "Left":
                df_heat = df_heat[df_heat["PitcherThrows"] == "Left"]

            # Apply Metric Filter
            if map_sel == "Whiffs":
                df_event = df_heat[df_heat["PitchCall"] == "StrikeSwinging"]
            elif map_sel == "Hard Hit (95+)":
                df_event = df_heat[df_heat["ExitSpeed"] >= 95]
            elif map_sel == "Softly Hit (<80)":
                df_event = df_heat[df_heat["ExitSpeed"] <= 80]
            elif map_sel == "Chases":
                swing_calls = ["strikeswinging", "foul", "inplay"]
                in_zone = df_heat["PlateLocSide"].between(-0.83, 0.83) & df_heat["PlateLocHeight"].between(1.5, 3.5)
                df_event = df_heat[df_heat["PitchCall"].str.lower().isin(swing_calls) & ~in_zone]
            elif map_sel == "Called Strikes":
                df_event = df_heat[df_heat["PitchCall"] == "StrikeCalled"]
            else: # All Pitches
                df_event = df_heat

            # Render Heatmaps
            if df_event.empty:
                st.warning(f"No data found for {map_sel} vs {pitcher_side_sel} Handed Pitchers.")
            else:
                families = ['Fastballs', 'Breaking', 'Off-Speed']
                cols = st.columns(3)

                for i, family in enumerate(families):
                    with cols[i]:
                        subset = df_event[df_event['PitchFamily'] == family]
                        
                        fig, ax = plt.subplots(figsize=(4, 5))
                        
                        ax.add_patch(Rectangle((-0.83, 1.5), 1.66, 2.0, 
                                            fill=False, edgecolor="black", linewidth=2.5, zorder=10))
                        ax.plot([-0.83, 0.83, 0.83, 0, -0.83, -0.83], [0, 0, 0.15, 0.3, 0.15, 0], color="black", lw=1.5)

                        if len(subset) < 5:
                            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, fontsize=12, color='gray')
                        else:
                            try:
                                sns.kdeplot(
                                    x=subset["PlateLocSide"],
                                    y=subset["PlateLocHeight"],
                                    fill=True,
                                    levels=10,
                                    thresh=0.05,
                                    bw_adjust=0.8,
                                    cmap="Reds",
                                    ax=ax,
                                    alpha=0.7
                                )
                            except:
                                ax.text(0.5, 0.5, "Low Density", ha="center", va="center", transform=ax.transAxes)

                        ax.set_xlim(-2.0, 2.0)
                        ax.set_ylim(0, 5)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(f"{family}\n(n={len(subset)})", fontsize=14, fontweight='bold')
                        
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)

                        st.pyplot(fig)
                        plt.close(fig)

elif app == "Holy Cross Pitcher":
    # --- HEADER SECTION ---
    st.subheader("✝️ Holy Cross Pitcher Analytics")

    # Hard-coded Team Selection
    selected_team_full = "College of the Holy Cross"
    
    # 1. Load Pitcher Names (Lazy Load)
    pitchers = load_players_for_team(selected_team_full, app_type="pitcher")
    
    # 2. Sidebar Selection
    # Format names: "O'Connor, Andrew" -> "Andrew O'Connor"
    pitchers_formatted = [' '.join(p.split(', ')[::-1]) if ', ' in p else p for p in pitchers]
    selected_pitcher_fmt = st.sidebar.selectbox("Select Holy Cross Pitcher", options=pitchers_formatted)
    
    # Convert back to raw format for data loading
    selected_pitcher_raw = ', '.join(selected_pitcher_fmt.split(' ')[::-1]) if ' ' in selected_pitcher_fmt else selected_pitcher_fmt
    
    # 3. Load Data (ONCE for all tabs)
    data = load_player_data(selected_pitcher_raw, selected_team_full, app_type="pitcher")

    # --- BIO HEADER ---
    if not data.empty:
        p_info = data.iloc[0]
        p_height = p_info['Height_Pitcher'] if pd.notnull(p_info['Height_Pitcher']) else "N/A"
        p_weight = f"{int(p_info['Weight_Pitcher'])} lbs" if pd.notnull(p_info['Weight_Pitcher']) else "N/A"
        p_jersey = f"#{int(p_info['Jersey_Pitcher'])}" if pd.notnull(p_info['Jersey_Pitcher']) else ""
        
        st.header(f"{selected_pitcher_fmt} {p_jersey}")
        st.subheader(f"{selected_team_full} • {p_info['PitcherThrows']}HP")
        st.write(f"**Physicals:** {p_height} | {p_weight}")
        st.divider()

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Performance Data", "Stuff Visuals", "Sequencing", "Usage", "Heatmaps", "Trends"])

    # --- TAB 1: PERFORMANCE DATA ---
    with tab1:
        if not data.empty:
            # Clean data
            df_tab1 = data[~data['TaggedPitchType'].isin(['Undefined', 'Other'])].copy()
            
            # Force numeric
            for col in ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'Extension']:
                df_tab1[col] = pd.to_numeric(df_tab1[col], errors='coerce')

            # Summary Table
            summary = df_tab1.groupby('TaggedPitchType').agg(
                Count=('TaggedPitchType', 'count'),
                AvgVelo=('RelSpeed', 'mean'),
                MaxVelo=('RelSpeed', 'max'),
                AvgSpin=('SpinRate', 'mean'),
                AvgIVB=('InducedVertBreak', 'mean'),
                AvgHB=('HorzBreak', 'mean'),
                AvgExt=('Extension', 'mean')
            ).reset_index()
            
            total_p = summary['Count'].sum()
            summary['Usage'] = (summary['Count'] / total_p * 100).round(1)
            summary = summary.rename(columns={'TaggedPitchType': 'Pitch', 'Count': '#'})
            
            st.subheader("Pitch Shape Summary")
            st.dataframe(
                summary[['Pitch', '#', 'Usage', 'AvgVelo', 'MaxVelo', 'AvgSpin', 'AvgIVB', 'AvgHB', 'AvgExt']].style.format({
                    'Usage': '{:.1f}%', 'AvgVelo': '{:.1f}', 'MaxVelo': '{:.1f}',
                    'AvgSpin': '{:.0f}', 'AvgIVB': '{:.1f}', 'AvgHB': '{:.1f}', 'AvgExt': '{:.2f}'
                }), use_container_width=True
            )

            # Splits
            for side in ['Left', 'Right']:
                st.subheader(f"vs {side}-Handed Hitters")
                split_df = df_tab1[df_tab1['BatterSide'] == side]
                if not split_df.empty:
                    stats_df = pitch_type_stats(split_df)
                    if not stats_df.empty:
                        overall_row = pd.DataFrame([overall_stats(split_df)])
                        final_df = pd.concat([overall_row, stats_df], ignore_index=True)
                        st.dataframe(final_df.style.format({
                            'Usage': '{:.1f}%', 'AVG': '{:.3f}', 'SLG': '{:.3f}',
                            'Zone%': '{:.1f}%', 'Whiff%': '{:.1f}%', 'Zone Whiff%': '{:.1f}%',
                            'Chase%': '{:.1f}%', 'run_value': '{:.2f}', 'wOBA': '{:.3f}',
                            'xwOBA': '{:.3f}', 'HH%': '{:.1f}%', 'GB%': '{:.1f}%'
                        }), use_container_width=True)
                else:
                    st.info(f"No pitch data found against {side}-handed hitters.")
        else:
            st.info("Select a pitcher to view data.")

    # --- TAB 2: STUFF VISUALS ---
    with tab2:
        # Filters (Date Only - NO Player Selector Here)
        c1, c2 = st.columns([1, 3]) 
        with c1:
            date_range = st.date_input("Date Range", [pd.to_datetime("2025-01-01"), pd.to_datetime("2025-12-31")], key="hc_p_date_tab2")

        if not data.empty:
            filtered_data = data.copy()
            filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
            
            # Numeric conversion for charts
            for col in ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'PlateLocSide', 'PlateLocHeight']:
                if col in filtered_data.columns:
                    filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

            # Apply Date Filter
            if len(date_range) == 2:
                filtered_data = filtered_data[(filtered_data['Date'] >= pd.to_datetime(date_range[0])) & 
                                              (filtered_data['Date'] <= pd.to_datetime(date_range[1]))]

            if filtered_data.empty:
                st.warning("No data found for this date range.")
            else:
                # --- LOCAL FIXING WIDGET ---
                def show_admin_fix_widget(selected_data, chart_name):
                    if not st.session_state.get("is_admin", False): return
                    if selected_data and "selection" in selected_data:
                        pts = selected_data["selection"]["points"]
                        if not pts: return
                        
                        safe_indices = [int(p["customdata"][0]) for p in pts if "customdata" in p]
                        
                        if safe_indices:
                            st.info(f"🔍 Selected {len(safe_indices)} pitches.")
                            c1, c2 = st.columns([2, 1])
                            with c1:
                                new_tag = st.selectbox("Change to:", list(PITCH_PALETTE.keys()), key=f"fix_{chart_name}")
                            with c2:
                                st.write("") 
                                if st.button(f"✅ Apply Fix", key=f"btn_{chart_name}"):
                                    try:
                                        path = get_local_data_path()
                                        full_df = pd.read_parquet(path)
                                        full_df.loc[safe_indices, 'TaggedPitchType'] = new_tag
                                        full_df.to_parquet("ncaa_data_2025_fixed.parquet", index=False)
                                        st.cache_data.clear()
                                        st.success("✅ Fixed! Download below.")
                                    except Exception as e: st.error(f"Error: {e}")

                # 1. MOVEMENT PROFILE
                st.subheader("Interactive Movement Profile (IVB vs HB)")
                st.caption("Lasso to inspect or fix pitch tags.")
                
                fig_mov = px.scatter(
                    filtered_data, x="HorzBreak", y="InducedVertBreak", 
                    color="TaggedPitchType", color_discrete_map=PITCH_PALETTE,
                    hover_data=["RelSpeed", "SpinRate", "Date"], width=650, height=650
                )
                fig_mov.add_hline(y=0, line_dash="dash", line_color="black")
                fig_mov.add_vline(x=0, line_dash="dash", line_color="black")
                fig_mov.update_layout(
                    xaxis=dict(range=[-30, 30], title="Horizontal Break (in)"), 
                    yaxis=dict(range=[-30, 30], title="Induced Vertical Break (in)", scaleanchor="x", scaleratio=1),
                    plot_bgcolor='white', dragmode='lasso'
                )
                fig_mov.update_traces(customdata=filtered_data.index) # Required for fixer

                selection_mov = st.plotly_chart(fig_mov, on_select="rerun", selection_mode=["box", "lasso"], key="hc_mov_scatter")
                show_admin_fix_widget(selection_mov, "hc_movement_chart")
                
                st.divider()

                # 2. VELOCITY vs SPIN RATE (The Missing Chart)
                st.subheader("Velocity vs. Spin Rate")
                st.caption("Identify misclassified pitches by spin/velo clusters.")

                fig_ss = px.scatter(
                    filtered_data, x='RelSpeed', y='SpinRate', 
                    color='TaggedPitchType', color_discrete_map=PITCH_PALETTE,
                    labels={'RelSpeed': 'Velocity (MPH)', 'SpinRate': 'Spin Rate (RPM)'},
                    hover_data=['RelSpeed', 'SpinRate', 'Date'], width=750, height=500
                )
                fig_ss.update_layout(plot_bgcolor='white', dragmode='lasso')
                fig_ss.update_traces(customdata=filtered_data.index, marker=dict(size=8, line=dict(width=1, color='white')))

                selection_ss = st.plotly_chart(fig_ss, on_select="rerun", selection_mode=["box", "lasso"], key="hc_velo_spin_scatter")
                show_admin_fix_widget(selection_ss, "hc_velo_spin_chart")

                st.divider()

                # 3. PERFORMANCE BY ZONE
                st.subheader("📍 Performance by Zone")
                z_col1, z_col2 = st.columns(2)
                with z_col1:
                    st.plotly_chart(draw_performance_grid(filtered_data, 'Left'), use_container_width=True)
                with z_col2:
                    st.plotly_chart(draw_performance_grid(filtered_data, 'Right'), use_container_width=True)

                # 4. ADMIN DOWNLOAD
                if st.session_state.get("is_admin", False):
                    st.divider()
                    st.markdown("### 💾 Save Your Work")
                    if os.path.exists("ncaa_data_2025_fixed.parquet"):
                        with open("ncaa_data_2025_fixed.parquet", "rb") as f:
                            st.download_button("⬇️ Download Fixed Data", f, "ncaa_data_2025.parquet", "application/octet-stream")
                    else:
                        st.info("Make a change above to generate a downloadable file.")
        else:
            st.info("Select a pitcher in the sidebar to view data.")

    # ... (Keep Tabs 3, 4, 5, 6 empty or with placeholder code if you haven't built them yet) ...


    with tab3:
        st.subheader(f"Pitch Sequencing")

        def pitch_sequencing_section(seq_data, label):
            st.markdown(f"### {label}")

            # Prepare data sorted by PA and pitch number
            seq_data = seq_data.sort_values(['Batter', 'Pitcher', 'PAofInning', 'PitchofPA'])

            # Identify previous pitch type for each pitch within a PA
            seq_data['PrevPitchType'] = seq_data.groupby(['Batter', 'Pitcher', 'PAofInning'])['TaggedPitchType'].shift(1)
            seq_data['Sequence'] = seq_data['PrevPitchType'] + '/' + seq_data['TaggedPitchType']

            # Filter out first pitches (no previous pitch)
            seq_data = seq_data[seq_data['PrevPitchType'].notnull()]

            total_sequences = len(seq_data)

            # Most common sequences (as percentage)
            common_seqs = seq_data['Sequence'].value_counts(normalize=True).reset_index()
            common_seqs.columns = ['Sequence', 'Percentage']
            common_seqs['Percentage'] = (common_seqs['Percentage'] * 100).round(1)
            st.write("**Most Common Sequences (%):**")
            st.dataframe(common_seqs.head(10))

            # Sequence outcomes
            def sequence_outcome_table(df, mask, label):
                filtered = df[mask]
                total = len(filtered)
                outcome_seqs = filtered['Sequence'].value_counts(normalize=True).reset_index()
                outcome_seqs.columns = ['Sequence', 'Percentage']
                outcome_seqs['Percentage'] = (outcome_seqs['Percentage'] * 100).round(1)
                st.write(f"**Sequences Leading to {label} (%):**")
                st.dataframe(outcome_seqs.head(10))

            # Whiffs: PitchCall == 'StrikeSwinging'
            whiff_mask = seq_data['PitchCall'] == 'StrikeSwinging'
            sequence_outcome_table(seq_data, whiff_mask, 'Whiffs')

            # Weak contact: ExitSpeed < 80 (only for pitches in play)
            weak_mask = (seq_data['PitchCall'] == 'InPlay') & (pd.to_numeric(seq_data['ExitSpeed'], errors='coerce') < 80)
            sequence_outcome_table(seq_data, weak_mask, 'Weak Contact (ExitSpeed < 80)')

            # Damage: ExitSpeed > 95 (only for pitches in play)
            damage_mask = (seq_data['PitchCall'] == 'InPlay') & (pd.to_numeric(seq_data['ExitSpeed'], errors='coerce') > 95)
            sequence_outcome_table(seq_data, damage_mask, 'Damage (ExitSpeed > 95)')

            # Count-specific sequences (e.g., 0-2, 1-2, 3-2) as percentage
            st.write("**Count-Specific Sequences (% for 0-2, 1-2, 3-2):**")
            seq_data['Count'] = seq_data['Balls'].astype(str) + '-' + seq_data['Strikes'].astype(str)
            for count in ['0-2', '1-2', '3-2']:
                count_seqs = seq_data[seq_data['Count'] == count]['Sequence'].value_counts(normalize=True).reset_index()
                count_seqs.columns = ['Sequence', 'Percentage']
                count_seqs['Percentage'] = (count_seqs['Percentage'] * 100).round(1)
                st.write(f"Most common sequences for count {count}:")
                st.dataframe(count_seqs.head(5))

        # Split by vLHH and vRHH
        vLHH_seq_data = data[data['BatterSide'] == 'Left']
        vRHH_seq_data = data[data['BatterSide'] == 'Right']

        pitch_sequencing_section(vLHH_seq_data, "vs Left-Handed Hitters (vLHH)")
        pitch_sequencing_section(vRHH_seq_data, "vs Right-Handed Hitters (vRHH)")

    with tab4:
        st.subheader(f"Pitch Usage Pie Charts")

        import matplotlib.pyplot as plt

        # Split data by vLHH and vRHH
        vLHH_data = data[data['BatterSide'] == 'Left']
        vRHH_data = data[data['BatterSide'] == 'Right']

        pitch_palette = {
            'Fastball': 'dodgerblue',
            'Curveball': 'red',
            'Slider': 'forestgreen',
            'Changeup': 'darkviolet',
            'Cutter': 'orange',
            'Sinker': 'gold',
            'Splitter': 'purple',
            'Knuckleball': 'black',
            'Other': 'grey'
        }

        # ---- Add legend at the top, only for used pitch types ----
        used_types = set(vLHH_data['TaggedPitchType'].unique()) | set(vRHH_data['TaggedPitchType'].unique())
        legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px;'>"
        for pitch, color in pitch_palette.items():
            if pitch in used_types:
                legend_html += f"<div style='display: flex; align-items: center; gap: 6px;'><div style='width: 16px; height: 16px; background: {color}; border-radius: 4px; border: 1px solid #888;'></div><span style='font-size: 14px;'>{pitch}</span></div>"
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

        def plot_pie_chart(df, mask_type):
            df = df.copy()
            df['Balls'] = pd.to_numeric(df['Balls'], errors='coerce').fillna(-1).astype(int)
            df['Strikes'] = pd.to_numeric(df['Strikes'], errors='coerce').fillna(-1).astype(int)
            if mask_type == "first":
                mask = (df['Balls'] == 0) & (df['Strikes'] == 0)
            elif mask_type == "hitter":
                mask = (df['Balls'] > df['Strikes'])
            elif mask_type == "pitcher":
                mask = (df['Strikes'] > df['Balls'])
            else:
                mask = pd.Series([True] * len(df))
            usage = df.loc[mask, 'TaggedPitchType'].value_counts()
            fig, ax = plt.subplots(figsize=(3, 3))  # All pies same size
            if not usage.empty:
                colors = [pitch_palette.get(pt, 'grey') for pt in usage.index]
                wedges, texts, autotexts = ax.pie(
                    usage,
                    labels=None,  # No labels on slices
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=140,
                    textprops={'fontsize': 12}
                )
                ax.axis('equal')  # Always a perfect circle
                for autotext in autotexts:
                    autotext.set_fontsize(12)
                ax.set_title("", fontsize=12)
                plt.tight_layout()
                return fig
            else:
                plt.close(fig)
                return None

        chart_types = [
            ("first", "First Pitch"),
            ("hitter", "Hitter's Count"),
            ("pitcher", "Pitcher's Ahead")
        ]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### vLHH")
        with col2:
            st.markdown("### vRHH")

        for mask_type, chart_title in chart_types:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{chart_title}**")
                fig = plot_pie_chart(vLHH_data, mask_type)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info(f"No pitch data available for vLHH {chart_title}.")
            with col2:
                st.markdown(f"**{chart_title}**")
                fig = plot_pie_chart(vRHH_data, mask_type)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info(f"No pitch data available for vRHH {chart_title}.")

    with tab5:
        st.header("Strike Zone Heatmaps")
        st.caption("Visualizing pitch density and location strategy.")

        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt
        import seaborn as sns

        # --- 1. DATA PREPARATION (Fixing the Numeric Error) ---
        df_heat = data.copy()

        # Force critical columns to numeric to prevent "categorical" errors
        plot_cols = ["PlateLocSide", "PlateLocHeight", "ExitSpeed"]
        for col in plot_cols:
            df_heat[col] = pd.to_numeric(df_heat[col], errors='coerce')

        # Drop rows without location data (KDE plots will fail without coordinates)
        df_heat = df_heat.dropna(subset=["PlateLocSide", "PlateLocHeight"])

        # --- 2. HEATMAP FILTERS ---
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            map_types = ["All Pitches", "Whiffs", "Hard Hit (95+)", "Softly Hit (<80)", "Chases", "Called Strikes"]
            map_sel = st.selectbox("Select Metric", map_types, key="hmap_metric_final")
        with h_col2:
            side_sel = st.radio("Batter Side", ["Combined", "Left", "Right"], horizontal=True, key="hmap_side_final")

        # Filter by Batter Side
        if side_sel != "Combined":
            df_heat = df_heat[df_heat["BatterSide"] == side_sel]

        # Filter by Event Type
        if map_sel == "All Pitches":
            df_event = df_heat
        elif map_sel == "Whiffs":
            df_event = df_heat[df_heat["PitchCall"] == "StrikeSwinging"]
        elif map_sel == "Hard Hit (95+)":
            df_event = df_heat[df_heat["ExitSpeed"] >= 95]
        elif map_sel == "Softly Hit (<80)":
            df_event = df_heat[df_heat["ExitSpeed"] <= 80]
        elif map_sel == "Chases":
            swing_calls = ["strikeswinging", "foul", "inplay"]
            # Strike Zone is roughly -0.83 to 0.83 horizontally, 1.5 to 3.5 vertically
            in_zone = df_heat["PlateLocSide"].between(-0.83, 0.83) & df_heat["PlateLocHeight"].between(1.5, 3.5)
            df_event = df_heat[df_heat["PitchCall"].str.lower().isin(swing_calls) & ~in_zone]
        else: # Called Strikes
            df_event = df_heat[df_heat["PitchCall"] == "StrikeCalled"]

        # --- 3. RENDER HEATMAPS ---
        if df_event.empty:
            st.warning(f"No data points found for {map_sel} vs {side_sel} hitters.")
        else:
            # Get top 5 pitch types by frequency
            top5_pitches = df_event["TaggedPitchType"].value_counts().index.tolist()[:5]
            
            # Create columns based on how many pitch types exist
            h_cols = st.columns(len(top5_pitches))
            
            for i, col in enumerate(h_cols):
                with col:
                    pt_type = top5_pitches[i]
                    subset = df_event[df_event["TaggedPitchType"] == pt_type]

                    # Create the Matplotlib Figure
                    fig, ax = plt.subplots(figsize=(4, 5))
                    
                    # Draw Strike Zone (Black Outline)
                    ax.add_patch(Rectangle((-0.83, 1.5), 1.66, 2.0, 
                                        fill=False, edgecolor="black", linewidth=2.5, zorder=10))
                    
                    # Draw Home Plate at the bottom for orientation
                    ax.plot([-0.83, 0.83, 0.83, 0, -0.83, -0.83], [0, 0, 0.15, 0.3, 0.15, 0], color="black", lw=1.5)

                    if len(subset) < 5:
                        ax.text(0.5, 0.5, "Not Enough\nData", ha="center", va="center", transform=ax.transAxes, fontsize=12)
                    else:
                        # Create Density Plot
                        sns.kdeplot(
                            x=subset["PlateLocSide"],
                            y=subset["PlateLocHeight"],
                            fill=True,
                            levels=10,
                            thresh=0.02,
                            bw_adjust=0.7,
                            cmap="Reds",
                            ax=ax,
                            alpha=0.7
                        )

                    # Visual Settings
                    ax.set_xlim(-2.5, 2.5) # Wide enough to see "Chase" pitches
                    ax.set_ylim(0, 5)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f"{pt_type}\n(n={len(subset)})", fontsize=14, fontweight='bold')
                    
                    st.pyplot(fig)
                    plt.close(fig) # Memory management

    with tab6:
        st.subheader("📈 Season Trends")
        st.caption("Track changes in velocity, spin, and movement profile game-by-game.")
        if not data.empty:
            plot_trend_lines(data)
        else:
            st.warning("No data available.")
        






