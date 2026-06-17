import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import pickle
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2026 World Cup Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; color: #111111; }
    section[data-testid="stSidebar"] { background-color: #1a1a3e; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a3e, #2d2d6e);
        border: 1px solid #4444aa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
        color: #ffffff;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #ccccff; margin-top: 4px; }
    .team-header {
        background: linear-gradient(135deg, #1a3a1a, #2d6e2d);
        border: 1px solid #44aa44;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 16px;
        color: #ffffff;
    }
    .vs-badge {
        font-size: 2.5rem; font-weight: bold;
        color: #ff6b35; text-align: center;
        padding: 30px 0; display: block;
    }
    .stButton > button {
        background-color: #1a1a3e !important;
        color: #ffffff !important;
        border: 2px solid #4444aa !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2d2d6e !important;
        border-color: #00d4ff !important;
    }
    label { color: #111111 !important; font-weight: 500; }
    .stRadio label { color: #111111 !important; }
    .stTabs [data-baseweb="tab"] { color: #111111 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    elo      = pd.read_csv("elo_all_teams_v2.csv")
    groups   = pd.read_csv("group_stages_clean.csv")
    fixtures = pd.read_csv("fixtures_clean.csv")
    ranks    = pd.read_csv("fifa_rankings.csv")
    sim_res  = pd.read_csv("simulation_results_v2.csv")
    return elo, groups, fixtures, ranks, sim_res

@st.cache_resource
def load_models():
    with open("home_model_v2.pkl", "rb") as f:
        home_model = pickle.load(f)
    with open("away_model_v2.pkl", "rb") as f:
        away_model = pickle.load(f)
    return home_model, away_model

elo_df, groups_df, fixtures_df, ranks_df, sim_results = load_data()
home_model, away_model = load_models()

# ── Team dictionary ───────────────────────────────────────────────────────────
team_info = elo_df.merge(ranks_df, left_on="team", right_on="Nation", how="left").drop(columns="Nation")
team_info = team_info.rename(columns={"FIFA_2026_rank": "fifa_rank"})
team_dict = team_info.set_index("team")[["elo", "fifa_rank"]].to_dict(orient="index")
for t in team_dict:
    if pd.isna(team_dict[t]["fifa_rank"]):
        team_dict[t]["fifa_rank"] = 100

# Add group info
group_map = dict(zip(groups_df["nation"], groups_df["group"]))
wc_teams  = groups_df["nation"].tolist()

# ── Dixon-Coles helpers ───────────────────────────────────────────────────────
RHO = -0.13

def dc_correction(i, j, hxg, axg, rho=RHO):
    if   i == 0 and j == 0: return 1 - hxg * axg * rho
    elif i == 1 and j == 0: return 1 + axg * rho
    elif i == 0 and j == 1: return 1 + hxg * rho
    elif i == 1 and j == 1: return 1 - rho
    return 1.0

def predict_match(home_team, away_team, neutral=True, tw=5.0):
    he  = team_dict[home_team]["elo"]
    ae  = team_dict[away_team]["elo"]
    hr  = team_dict[home_team]["fifa_rank"]
    ar  = team_dict[away_team]["fifa_rank"]
    row = pd.DataFrame([{"home_elo": he, "away_elo": ae, "elo_diff": he-ae,
                          "neutral": int(neutral), "tournament_weight": tw,
                          "fifa_rank_diff": hr - ar}])
    return home_model.predict(row)[0], away_model.predict(row)[0]

def match_probabilities(hxg, axg, max_goals=10):
    hw, d, aw = 0, 0, 0
    score_grid = {}
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, hxg) * poisson.pmf(j, axg) * dc_correction(i, j, hxg, axg)
            score_grid[(i, j)] = p
            if i > j: hw += p
            elif i == j: d += p
            else: aw += p
    return hw, d, aw, score_grid

def simulate_match(ht, at, neutral=True, tw=5.0):
    hxg, axg = predict_match(ht, at, neutral, tw)
    return np.random.poisson(hxg), np.random.poisson(axg)

def simulate_knockout(t1, t2):
    hxg, axg = predict_match(t1, t2, neutral=True)
    hg = np.random.poisson(hxg)
    ag = np.random.poisson(axg)
    if hg == ag:
        hg += np.random.poisson(hxg * 0.33)
        ag += np.random.poisson(axg * 0.33)
    if hg == ag:
        pen = np.clip(0.5 + (team_dict[t1]["elo"] - team_dict[t2]["elo"]) * 0.0001, 0.4, 0.6)
        if np.random.random() < pen: hg += 1
        else: ag += 1
    return (t1 if hg > ag else t2), hg, ag

def simulate_group(group_name):
    teams = groups_df[groups_df["group"] == group_name]["nation"].tolist()
    table = {t: {"W":0,"D":0,"L":0,"GF":0,"GA":0,"Pts":0} for t in teams}
    gf = fixtures_df[fixtures_df["home_team"].isin(teams) & fixtures_df["away_team"].isin(teams)]
    for _, row in gf.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = simulate_match(h, a, neutral=row.get("neutral", True))
        table[h]["GF"]+=hg; table[h]["GA"]+=ag
        table[a]["GF"]+=ag; table[a]["GA"]+=hg
        if hg>ag: table[h]["W"]+=1; table[h]["Pts"]+=3; table[a]["L"]+=1
        elif hg<ag: table[a]["W"]+=1; table[a]["Pts"]+=3; table[h]["L"]+=1
        else: table[h]["D"]+=1; table[a]["D"]+=1; table[h]["Pts"]+=1; table[a]["Pts"]+=1
    tdf = pd.DataFrame(table).T
    tdf["GD"] = tdf["GF"] - tdf["GA"]
    return tdf.sort_values(["Pts","GD","GF"], ascending=False).reset_index().rename(columns={"index":"Team"})

def simulate_group_live(group_name, completed_lookup):
    teams = groups_df[groups_df["group"] == group_name]["nation"].tolist()
    table = {t: {"W":0,"D":0,"L":0,"GF":0,"GA":0,"Pts":0} for t in teams}
    gf = fixtures_df[fixtures_df["home_team"].isin(teams) & fixtures_df["away_team"].isin(teams)]
    for _, row in gf.iterrows():
        h, a = row["home_team"], row["away_team"]
        pair = frozenset({h, a})
        if pair in completed_lookup:
            ch, ca, chs, cas = completed_lookup[pair]
            if ch == h: hg, ag = chs, cas
            else: hg, ag = cas, chs
        else:
            hg, ag = simulate_match(h, a, neutral=row.get("neutral", True))
        table[h]["GF"]+=hg; table[h]["GA"]+=ag
        table[a]["GF"]+=ag; table[a]["GA"]+=hg
        if hg>ag: table[h]["W"]+=1; table[h]["Pts"]+=3; table[a]["L"]+=1
        elif hg<ag: table[a]["W"]+=1; table[a]["Pts"]+=3; table[h]["L"]+=1
        else: table[h]["D"]+=1; table[a]["D"]+=1; table[h]["Pts"]+=1; table[a]["Pts"]+=1
    tdf = pd.DataFrame(table).T
    tdf["GD"] = tdf["GF"] - tdf["GA"]
    return tdf.sort_values(["Pts","GD","GF"], ascending=False).reset_index().rename(columns={"index":"Team"})

def get_qualifiers_live(completed_lookup):
    winners, runners, thirds = [], [], []
    for g in sorted(groups_df["group"].unique()):
        t = simulate_group_live(g, completed_lookup)
        winners.append(t.iloc[0]["Team"])
        runners.append(t.iloc[1]["Team"])
        thirds.append({"team": t.iloc[2]["Team"], "pts": t.iloc[2]["Pts"],
                       "gd": t.iloc[2]["GD"], "gf": t.iloc[2]["GF"]})
    best_thirds = pd.DataFrame(thirds).sort_values(["pts","gd","gf"], ascending=False).head(8)["team"].tolist()
    return winners + runners + best_thirds

def parse_results_text(text, fixtures_df, team_dict):
    """Parse lines like 'Spain 3-0 Morocco' into completed result dicts + matched fixture lookup."""
    import re
    pattern = re.compile(r'^\s*(.+?)\s+(\d+)\s*[-–:]\s*(\d+)\s+(.+?)\s*$')
    parsed, errors = [], []

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            errors.append((line, "Could not parse — use format 'TeamA 2-1 TeamB'"))
            continue
        t1, s1, t2_score, t2 = m.group(1).strip(), int(m.group(2)), int(m.group(3)), m.group(4).strip()
        if t1 not in team_dict:
            errors.append((line, f"Unknown team: '{t1}'"))
            continue
        if t2 not in team_dict:
            errors.append((line, f"Unknown team: '{t2}'"))
            continue
        parsed.append({"home_team": t1, "away_team": t2, "home_score": s1, "away_score": t2_score})

    completed_lookup = {}
    matched, unmatched = [], []
    for res in parsed:
        ht, at, hs, as_ = res["home_team"], res["away_team"], res["home_score"], res["away_score"]
        pair = frozenset({ht, at})
        fixture_match = fixtures_df[
            fixtures_df.apply(lambda r: frozenset({r["home_team"], r["away_team"]}) == pair, axis=1)
        ]
        if len(fixture_match) == 0:
            unmatched.append(res)
            continue
        completed_lookup[pair] = (ht, at, hs, as_)
        matched.append(res)

    return completed_lookup, matched, unmatched, errors


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/3/35/2026_FIFA_World_Cup.svg/200px-2026_FIFA_World_Cup.svg.png", width=120)
st.sidebar.title("2026 World Cup")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏆 Champion Odds",
    "⚔️ Head-to-Head",
    "📊 Group Stage",
    "🏟️ Bracket Simulator",
    "👤 Team Profile",
    "🔴 Live Tracker",
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Champion Odds
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏆 Champion Odds":
    st.title("🏆 2026 World Cup — Champion Probabilities")
    st.caption("Based on 10,000 Monte Carlo simulations using Elo formula")

    col1, col2 = st.columns([3, 1])

    with col2:
        top_n = st.slider("Show top N teams", 10, 48, 20)
        stage = st.selectbox("Stage", ["Winner %", "Final %", "SF %", "QF %", "R16 %", "R32 %"])

    plot_df = sim_results.sort_values(stage, ascending=False).head(top_n)

    with col1:
        fig = px.bar(
            plot_df, x="Team", y=stage,
            color=stage,
            color_continuous_scale="Blues",
            title=f"Top {top_n} Teams — {stage}",
            labels={stage: "Probability (%)"},
            height=500
        )
        fig.update_layout(
            plot_bgcolor="#0a0a1a", paper_bgcolor="#0a0a1a",
            font_color="white", showlegend=False,
            xaxis_tickangle=-45,
            coloraxis_showscale=False,
        )
        fig.update_traces(marker_line_color="#00d4ff", marker_line_width=0.5)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Probability Table")
    display_df = sim_results.copy()
    display_df.index = range(1, len(display_df)+1)
    st.dataframe(display_df, use_container_width=True, height=600)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Head-to-Head
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Head-to-Head":
    st.title("⚔️ Head-to-Head Match Predictor")

    col1, col_vs, col2 = st.columns([2, 1, 2])

    with col1:
        team1 = st.selectbox("Home Team", sorted(wc_teams), index=sorted(wc_teams).index("Spain"))
    with col_vs:
        st.markdown("<div class='vs-badge'>VS</div>", unsafe_allow_html=True)
    with col2:
        team2 = st.selectbox("Away Team", sorted(wc_teams), index=sorted(wc_teams).index("Argentina"))

    neutral = st.checkbox("Neutral venue (default for WC)", value=True)

    if team1 == team2:
        st.warning("Please select two different teams.")
    else:
        hxg, axg = predict_match(team1, team2, neutral=neutral)
        hw, d, aw, score_grid = match_probabilities(hxg, axg)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{hw*100:.1f}%</div>
                <div class='metric-label'>{team1} Win</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{d*100:.1f}%</div>
                <div class='metric-label'>Draw</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{aw*100:.1f}%</div>
                <div class='metric-label'>{team2} Win</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Expected Goals")
            fig_xg = go.Figure(go.Bar(
                x=[team1, team2], y=[round(hxg, 2), round(axg, 2)],
                marker_color=["#00d4ff", "#ff6b35"],
                text=[f"{hxg:.2f}", f"{axg:.2f}"], textposition="auto"
            ))
            fig_xg.update_layout(
                plot_bgcolor="#0a0a1a", paper_bgcolor="#0a0a1a",
                font_color="white", showlegend=False, height=300
            )
            st.plotly_chart(fig_xg, use_container_width=True)

        with col_b:
            st.subheader("Most Likely Scorelines")
            score_df = pd.DataFrame([
                {"Score": f"{i}-{j}", "Probability": f"{v*100:.1f}%", "_p": v}
                for (i, j), v in score_grid.items()
            ]).sort_values("_p", ascending=False).head(8).drop(columns="_p")
            score_df.index = range(1, len(score_df)+1)
            st.dataframe(score_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Team Comparison")
        tc1, tc2 = st.columns(2)
        with tc1:
            e1 = team_dict[team1]["elo"]
            r1 = team_dict[team1]["fifa_rank"]
            g1 = group_map.get(team1, "N/A")
            s1 = sim_results[sim_results["Team"] == team1]
            st.markdown(f"""<div class='team-header'>
                <b style='font-size:1.3rem'>{team1}</b><br>
                Elo: <b>{e1:.0f}</b> &nbsp;|&nbsp; FIFA Rank: <b>#{int(r1)}</b> &nbsp;|&nbsp; Group: <b>{g1}</b><br>
                Win probability: <b>{s1['Winner %'].values[0] if len(s1) else 'N/A'}%</b>
            </div>""", unsafe_allow_html=True)
        with tc2:
            e2 = team_dict[team2]["elo"]
            r2 = team_dict[team2]["fifa_rank"]
            g2 = group_map.get(team2, "N/A")
            s2 = sim_results[sim_results["Team"] == team2]
            st.markdown(f"""<div class='team-header'>
                <b style='font-size:1.3rem'>{team2}</b><br>
                Elo: <b>{e2:.0f}</b> &nbsp;|&nbsp; FIFA Rank: <b>#{int(r2)}</b> &nbsp;|&nbsp; Group: <b>{g2}</b><br>
                Win probability: <b>{s2['Winner %'].values[0] if len(s2) else 'N/A'}%</b>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Group Stage
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Group Stage":
    st.title("📊 Group Stage — Probabilities & Simulation")

    tab1, tab2 = st.tabs(["📈 Qualification Odds", "🎲 Simulate a Group"])

    with tab1:
        all_groups = sorted(groups_df["group"].unique())
        cols = st.columns(2)
        for idx, grp in enumerate(all_groups):
            with cols[idx % 2]:
                st.subheader(f"Group {grp}")
                grp_teams = groups_df[groups_df["group"] == grp]["nation"].tolist()
                grp_data = []
                for t in grp_teams:
                    row = sim_results[sim_results["Team"] == t]
                    grp_data.append({
                        "Team": t,
                        "Elo": round(team_dict[t]["elo"]),
                        "FIFA Rank": int(team_dict[t]["fifa_rank"]),
                        "R32 %": row["R32 %"].values[0] if len(row) else 0,
                        "R16 %": row["R16 %"].values[0] if len(row) else 0,
                        "QF %":  row["QF %"].values[0]  if len(row) else 0,
                        "Win %": row["Winner %"].values[0] if len(row) else 0,
                    })
                grp_df = pd.DataFrame(grp_data).sort_values("Elo", ascending=False).reset_index(drop=True)
                grp_df.index += 1
                st.dataframe(grp_df, use_container_width=True)

    with tab2:
        grp_select = st.selectbox("Select group to simulate", sorted(groups_df["group"].unique()))
        n_sims = st.slider("Number of simulations", 100, 5000, 1000)

        if st.button("🎲 Run Simulation", key="grp_sim"):
            grp_teams = groups_df[groups_df["group"] == grp_select]["nation"].tolist()
            finish_counts = defaultdict(lambda: defaultdict(int))

            with st.spinner(f"Simulating Group {grp_select} {n_sims:,} times..."):
                for _ in range(n_sims):
                    t = simulate_group(grp_select)
                    for pos, row in t.iterrows():
                        finish_counts[row["Team"]][pos+1] += 1

            results = []
            for team in grp_teams:
                results.append({
                    "Team": team,
                    "1st %": round(finish_counts[team][1] / n_sims * 100, 1),
                    "2nd %": round(finish_counts[team][2] / n_sims * 100, 1),
                    "3rd %": round(finish_counts[team][3] / n_sims * 100, 1),
                    "4th %": round(finish_counts[team][4] / n_sims * 100, 1),
                })

            res_df = pd.DataFrame(results).sort_values("1st %", ascending=False).reset_index(drop=True)
            res_df.index += 1
            st.success(f"Done! Results from {n_sims:,} simulations of Group {grp_select}")
            st.dataframe(res_df, use_container_width=True)

            fig = px.bar(
                res_df.melt(id_vars="Team", var_name="Position", value_name="Probability"),
                x="Team", y="Probability", color="Position",
                barmode="group", height=400,
                color_discrete_map={"1st %":"#ffd700","2nd %":"#c0c0c0","3rd %":"#cd7f32","4th %":"#666666"}
            )
            fig.update_layout(plot_bgcolor="#0a0a1a", paper_bgcolor="#0a0a1a", font_color="white")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Bracket Simulator
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏟️ Bracket Simulator":
    st.title("🏟️ Bracket Simulator")

    mode = st.radio("Mode", ["🤖 Auto-simulate (model picks)", "✋ Manual (you pick)"], horizontal=True)
    st.markdown("---")

    if mode == "🤖 Auto-simulate (model picks)":
        if st.button("▶️ Run Full Tournament"):
            with st.spinner("Simulating tournament..."):
                # Group stage
                all_tables = {}
                for grp in sorted(groups_df["group"].unique()):
                    all_tables[grp] = simulate_group(grp)

                winners, runners, thirds = [], [], []
                for grp, table in all_tables.items():
                    winners.append(table.iloc[0]["Team"])
                    runners.append(table.iloc[1]["Team"])
                    thirds.append({"team": table.iloc[2]["Team"],
                                   "pts": table.iloc[2]["Pts"],
                                   "gd":  table.iloc[2]["GD"],
                                   "gf":  table.iloc[2]["GF"]})

                best_thirds = pd.DataFrame(thirds).sort_values(["pts","gd","gf"], ascending=False).head(8)["team"].tolist()
                qualifiers  = winners + runners + best_thirds

            # Show group results
            st.subheader("Group Stage Results")
            gcols = st.columns(4)
            for idx, (grp, table) in enumerate(all_tables.items()):
                with gcols[idx % 4]:
                    st.markdown(f"**Group {grp}**")
                    mini = table[["Team","Pts","GD"]].copy()
                    mini.index = range(1, 5)
                    st.dataframe(mini, use_container_width=True, height=175)

            st.markdown("---")
            st.subheader("Knockout Rounds")

            current_round = qualifiers.copy()
            round_names   = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
            champion      = None

            for round_name in round_names:
                st.markdown(f"**{round_name}**")
                np.random.shuffle(current_round)
                next_round = []
                rcols = st.columns(min(4, len(current_round)//2))
                for i in range(0, len(current_round), 2):
                    t1, t2 = current_round[i], current_round[i+1]
                    winner, hg, ag = simulate_knockout(t1, t2)
                    next_round.append(winner)
                    col_idx = (i//2) % len(rcols)
                    with rcols[col_idx]:
                        color = "#00d4ff" if winner == t1 else "#ff6b35"
                        st.markdown(f"<small>**{t1}** {hg}–{ag} **{t2}** → <span style='color:{color}'>{winner}</span></small>", unsafe_allow_html=True)
                current_round = next_round
                st.markdown("")

            champion = current_round[0]
            st.balloons()
            st.markdown(f"<div style='text-align:center; font-size:2.5rem; padding:30px;'>🏆 <b>{champion}</b> wins the 2026 World Cup!</div>", unsafe_allow_html=True)

    else:  # Manual mode
        st.info("First, simulate the group stage automatically, then pick winners manually in each knockout round.")

        if st.button("🎲 Simulate Group Stage"):
            all_tables = {}
            for grp in sorted(groups_df["group"].unique()):
                all_tables[grp] = simulate_group(grp)

            winners, runners, thirds = [], [], []
            for grp, table in all_tables.items():
                winners.append(table.iloc[0]["Team"])
                runners.append(table.iloc[1]["Team"])
                thirds.append({"team": table.iloc[2]["Team"],
                               "pts": table.iloc[2]["Pts"],
                               "gd":  table.iloc[2]["GD"],
                               "gf":  table.iloc[2]["GF"]})

            best_thirds = pd.DataFrame(thirds).sort_values(["pts","gd","gf"], ascending=False).head(8)["team"].tolist()
            st.session_state["qualifiers"] = winners + runners + best_thirds
            st.session_state["group_tables"] = all_tables

        if "group_tables" in st.session_state:
            gcols = st.columns(4)
            for idx, (grp, table) in enumerate(st.session_state["group_tables"].items()):
                with gcols[idx % 4]:
                    st.markdown(f"**Group {grp}**")
                    mini = table[["Team","Pts","GD"]].copy()
                    mini.index = range(1, 5)
                    st.dataframe(mini, use_container_width=True, height=175)

        if "qualifiers" in st.session_state:
            st.markdown("---")
            st.subheader("Pick Your Winners")
            qualifiers = st.session_state["qualifiers"].copy()
            current_round = qualifiers.copy()
            round_names   = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
            picks = {}

            for round_name in round_names:
                if len(current_round) < 2: break
                st.markdown(f"**{round_name}**")
                next_round = []
                rcols = st.columns(min(4, len(current_round)//2))
                for i in range(0, len(current_round), 2):
                    t1, t2 = current_round[i], current_round[i+1]
                    hw, d, aw, _ = match_probabilities(*predict_match(t1, t2))
                    col_idx = (i//2) % len(rcols)
                    with rcols[col_idx]:
                        pick = st.radio(
                            f"{t1} ({hw*100:.0f}%) vs {t2} ({aw*100:.0f}%)",
                            [t1, t2], key=f"{round_name}_{i}"
                        )
                        next_round.append(pick)
                current_round = next_round

            if len(current_round) == 1:
                champion = current_round[0]
                st.balloons()
                st.markdown(f"<div style='text-align:center; font-size:2.5rem; padding:30px;'>🏆 <b>{champion}</b> wins the 2026 World Cup!</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Team Profile
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Team Profile":
    st.title("👤 Team Profile")

    team = st.selectbox("Select a team", sorted(wc_teams))
    st.markdown("---")

    elo_val  = team_dict[team]["elo"]
    rank_val = int(team_dict[team]["fifa_rank"])
    grp      = group_map.get(team, "N/A")
    sim_row  = sim_results[sim_results["Team"] == team]

    # Header
    elo_rank = elo_df.sort_values("elo", ascending=False).reset_index(drop=True)
    elo_rank.index += 1
    elo_position = elo_rank[elo_rank["team"] == team].index[0] if team in elo_rank["team"].values else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{elo_val:.0f}</div>
            <div class='metric-label'>Elo Rating</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>#{elo_position}</div>
            <div class='metric-label'>Elo Rank (all teams)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>#{rank_val}</div>
            <div class='metric-label'>FIFA Rank</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{grp}</div>
            <div class='metric-label'>Group</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if len(sim_row) > 0:
        st.subheader("Tournament Stage Probabilities")
        stages    = ["R32 %", "R16 %", "QF %", "SF %", "Final %", "Winner %"]
        stage_labels = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final", "Champion"]
        probs = [sim_row[s].values[0] for s in stages]

        fig = go.Figure(go.Bar(
            x=stage_labels, y=probs,
            marker_color=["#1e3a5f","#1e5f3a","#5f5f1e","#5f1e5f","#5f1e1e","#ffd700"],
            text=[f"{p}%" for p in probs], textposition="auto",
        ))
        fig.update_layout(
            plot_bgcolor="#0a0a1a", paper_bgcolor="#0a0a1a",
            font_color="white", height=350, showlegend=False,
            yaxis_title="Probability (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Group {grp} — Opponents")
    grp_teams = groups_df[groups_df["group"] == grp]["nation"].tolist()
    opponents = [t for t in grp_teams if t != team]

    ocols = st.columns(len(opponents))
    for idx, opp in enumerate(opponents):
        with ocols[idx]:
            hxg, axg = predict_match(team, opp)
            hw, d, aw, _ = match_probabilities(hxg, axg)
            opp_elo = team_dict[opp]["elo"]
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'><b>{opp}</b></div>
                <div class='metric-label'>Elo: {opp_elo:.0f}</div>
                <div style='margin-top:8px; font-size:0.8rem; color:#aaa'>
                    Win {hw*100:.0f}% | Draw {d*100:.0f}% | Loss {aw*100:.0f}%
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Live Tracker
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔴 Live Tracker":
    st.title("🔴 Live Tracker — 2026 World Cup")
    st.caption("Paste completed match results to update simulation odds. Your original pre-tournament prediction stays unchanged.")

    if "completed_results_text" not in st.session_state:
        st.session_state["completed_results_text"] = ""

    st.subheader("Enter Completed Results")
    st.markdown("One match per line, format: **`TeamA 2-1 TeamB`**")

    results_text = st.text_area(
        "Completed matches",
        value=st.session_state["completed_results_text"],
        height=150,
        placeholder="Spain 3-0 Morocco\nArgentina 1-1 Brazil\nFrance 2-0 Senegal"
    )
    st.session_state["completed_results_text"] = results_text

    n_sims_live = st.slider("Number of simulations", 500, 10000, 2000, step=500)

    if st.button("🔄 Update Simulation with Live Results"):
        if not results_text.strip():
            st.warning("Enter at least one completed match result.")
        else:
            completed_lookup, matched, unmatched, errors = parse_results_text(results_text, fixtures_df, team_dict)

            if errors:
                st.error("Some lines couldn't be parsed:")
                for line, msg in errors:
                    st.markdown(f"- `{line}` — {msg}")

            if unmatched:
                st.warning("These results don't match any group stage fixture (possibly already a knockout match — not yet supported):")
                for r in unmatched:
                    st.markdown(f"- {r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']}")

            if matched:
                st.success(f"✅ Matched {len(matched)} completed result(s)")

                st.markdown("---")
                st.subheader("Current Group Standings")
                st.caption("Completed matches use real scores; remaining matches show one sample simulation")

                affected_groups = set()
                for ht, at, _, _ in completed_lookup.values():
                    if ht in group_map: affected_groups.add(group_map[ht])
                    if at in group_map: affected_groups.add(group_map[at])

                gcols = st.columns(4)
                for idx, grp in enumerate(sorted(affected_groups)):
                    with gcols[idx % 4]:
                        st.markdown(f"**Group {grp}**")
                        t = simulate_group_live(grp, completed_lookup)
                        mini = t[["Team","Pts","GD","GF","GA"]].copy()
                        mini.index = range(1, 5)
                        st.dataframe(mini, use_container_width=True, height=175)

                st.markdown("---")
                st.subheader("Updated Champion Probabilities")

                with st.spinner(f"Running {n_sims_live:,} live simulations..."):
                    stage_counts = defaultdict(lambda: defaultdict(int))
                    for _ in range(n_sims_live):
                        qualifiers = get_qualifiers_live(completed_lookup)
                        for team in qualifiers:
                            stage_counts[team]['R32'] += 1
                        current_round = qualifiers.copy()
                        round_names = ['R16','QF','SF','Final']
                        np.random.shuffle(current_round)
                        for rn in round_names:
                            next_round = []
                            for i in range(0, len(current_round), 2):
                                t1, t2 = current_round[i], current_round[i+1]
                                winner, _, _ = simulate_knockout(t1, t2)
                                next_round.append(winner)
                                stage_counts[winner][rn] += 1
                            current_round = next_round
                        stage_counts[current_round[0]]['Winner'] += 1

                live_results = []
                for team in wc_teams:
                    live_results.append({
                        "Team": team,
                        "R32 %": round(stage_counts[team]['R32'] / n_sims_live * 100, 1),
                        "R16 %": round(stage_counts[team]['R16'] / n_sims_live * 100, 1),
                        "QF %":  round(stage_counts[team]['QF']  / n_sims_live * 100, 1),
                        "SF %":  round(stage_counts[team]['SF']  / n_sims_live * 100, 1),
                        "Final %": round(stage_counts[team]['Final'] / n_sims_live * 100, 1),
                        "Winner %": round(stage_counts[team]['Winner'] / n_sims_live * 100, 1),
                    })

                live_df = pd.DataFrame(live_results).sort_values("Winner %", ascending=False).reset_index(drop=True)
                live_df.index += 1

                comparison = live_df.merge(
                    sim_results[["Team","Winner %"]].rename(columns={"Winner %":"Winner % (Pre-tournament)"}),
                    on="Team", how="left"
                )
                comparison["Change"] = (comparison["Winner %"] - comparison["Winner % (Pre-tournament)"]).round(1)
                comparison = comparison[["Team","Winner % (Pre-tournament)","Winner %","Change","R32 %","R16 %","QF %","SF %","Final %"]]
                comparison = comparison.rename(columns={"Winner %": "Winner % (Live)"})
                comparison.index = range(1, len(comparison)+1)

                st.dataframe(comparison.head(20), use_container_width=True, height=500)

                st.markdown("---")
                if st.button("💾 Save as simulation_results_live.csv"):
                    save_path = "simulation_results_live.csv"
                    live_df.to_csv(save_path, index=False)
                    st.success(f"Saved to {save_path}")