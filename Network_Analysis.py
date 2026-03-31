"""
============================================================
  UBO / KYC Due Diligence Dashboard
  Companies House API + Panama Papers + FATF Screening
  PostgreSQL Customer Database
============================================================
  Run:  py -3.11 CH_Panama_Analyst_Dashboard.py
  Open: http://127.0.0.1:8050
============================================================
"""

import requests
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
from datetime import datetime, date

# ============================================================
# CONFIG
# ============================================================
CH_API_KEY = ##GENERATED API KEY##

DB_CONFIG = {
    "host":     "localhost",
    "port":     ##DEFINED PORT##,
    "database": "fi_customers",
    "user":     "postgres",
    "password": #DEFINED PGSQL PW##
}

PANAMA_PATHS = [
    ##DEFINED_DATABASE_DIRECTORY##,
]

BASE_URL = "https://api.companieshouse.gov.uk"

# ============================================================
# FATF Grey and Black Lists (as of 2025)
# ============================================================
FATF_BLACK_LIST = {
    "iran", "north korea", "myanmar", "russia"
}

FATF_GREY_LIST = {
    "albania", "barbados", "burkina faso", "cameroon", "cayman islands",
    "congo", "croatia", "democratic republic of congo", "ethiopia",
    "haiti", "jamaica", "jordan", "kenya", "laos", "mali", "mozambique",
    "namibia", "nigeria", "philippines", "senegal", "south africa",
    "south sudan", "syria", "tanzania", "trinidad and tobago",
    "uganda", "united arab emirates", "uae", "vietnam", "yemen"
}

HIGH_RISK_JURISDICTIONS = {
    "panama", "british virgin islands", "bvi", "cayman islands",
    "seychelles", "isle of man", "jersey", "guernsey", "belize",
    "bahamas", "marshall islands", "vanuatu", "samoa", "niue",
    "liechtenstein", "andorra", "monaco"
}

def jurisdiction_risk(country_raw: str):
    """
    Returns (risk_label, severity) for a given country string.
    Checks FATF black, grey, and high-risk jurisdiction lists.
    """
    c = str(country_raw).strip().lower()
    if not c or c == "nan":
        return None, None
    if any(b in c for b in FATF_BLACK_LIST):
        return "FATF Black List", "Critical"
    if any(g in c for g in FATF_GREY_LIST):
        return "FATF Grey List", "High"
    if any(h in c for h in HIGH_RISK_JURISDICTIONS):
        return "High-Risk Jurisdiction", "High"
    return None, None

# ============================================================
# 1. Load Panama Papers index
# ============================================================
print("Loading Panama Papers index...")
panama_frames = []
for path in PANAMA_PATHS:
    try:
        df = pd.read_csv(path, usecols=["n.name"], low_memory=False)
        panama_frames.append(df)
    except Exception as e:
        print(f"  Warning: {path} — {e}")

if panama_frames:
    panama_df = pd.concat(panama_frames, ignore_index=True)
    panama_name_set = set(
        panama_df["n.name"].dropna().str.strip().str.lower().unique()
    )
    print(f"  Loaded {len(panama_name_set):,} Panama names.")
else:
    panama_name_set = set()

def panama_hit(name):
    return str(name).strip().lower() in panama_name_set

# ============================================================
# 2. Database helpers
# ============================================================
def get_db_conn():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

def db_query(sql, params=None):
    try:
        conn = get_db_conn()
        cur  = conn.cursor()
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"DB error: {e}")
        return []

def load_company_options():
    rows = db_query("""
        SELECT DISTINCT ON (c.company_number)
               c.company_number, c.company_name
        FROM companies c
        ORDER BY c.company_number, c.company_name
    """)
    return [
        {"label": f"{r['company_name']} ({r['company_number']})",
         "value": r["company_number"]}
        for r in rows
    ]

# ============================================================
# 3. Companies House API helpers
# ============================================================
def ch_get(endpoint):
    url  = BASE_URL + endpoint
    resp = requests.get(url, auth=(CH_API_KEY, ""), timeout=10)
    return resp.json() if resp.status_code == 200 else {}

def get_officers(number):
    return ch_get(f"/company/{number}/officers?items_per_page=50").get("items", [])

def get_pscs(number):
    return ch_get(
        f"/company/{number}/persons-with-significant-control?items_per_page=50"
    ).get("items", [])

def get_filings(number):
    return ch_get(f"/company/{number}/filing-history?items_per_page=15").get("items", [])

# ============================================================
# 4. Risk Scoring Engine
# ============================================================
def compute_risk(profile, officers, pscs, filings):
    """
    Returns dict with score, tier, color, and detailed flags.
    Each flag has: flag text, severity, category.
    """
    flags  = []
    score  = 0

    # --- Company status ---
    status = profile.get("company_status", "").lower()
    if status in ("dissolved", "liquidation", "receivership", "administration"):
        flags.append({"flag": f"Company status: {status.title()}",
                      "severity": "High", "category": "Company Status"})
        score += 25
    elif status not in ("active", "open"):
        flags.append({"flag": f"Non-standard status: {status}",
                      "severity": "Medium", "category": "Company Status"})
        score += 10

    # --- Registered address jurisdiction ---
    addr    = profile.get("registered_office_address", {})
    country = addr.get("country", "")
    jx_label, jx_sev = jurisdiction_risk(country)
    if jx_label:
        flags.append({
            "flag": f"Registered address in {jx_label}: {country}",
            "severity": jx_sev,
            "category": "Jurisdiction — Registered Address"
        })
        score += 30 if jx_sev == "Critical" else 20

    # --- Officer nationalities / countries of residence ---
    for o in officers:
        for field in ["nationality", "country_of_residence"]:
            val = o.get(field, "")
            jl, js = jurisdiction_risk(val)
            if jl:
                flags.append({
                    "flag": f"Officer {o.get('name','')} — {field.replace('_',' ').title()}: {val} ({jl})",
                    "severity": js,
                    "category": "Jurisdiction — Officer"
                })
                score += 20 if js == "Critical" else 10
                break

    # --- PSC nationalities / countries of residence ---
    for p in pscs:
        for field in ["nationality", "country_of_residence"]:
            val = p.get(field, "")
            jl, js = jurisdiction_risk(val)
            if jl:
                name = p.get("name", "Unknown PSC")
                flags.append({
                    "flag": f"PSC {name} — {field.replace('_',' ').title()}: {val} ({jl})",
                    "severity": js,
                    "category": "Jurisdiction — PSC"
                })
                score += 20 if js == "Critical" else 10
                break

    # --- No PSCs ---
    if not pscs:
        flags.append({"flag": "No Persons with Significant Control registered — ownership opaque",
                      "severity": "High", "category": "Ownership Structure"})
        score += 20

    # --- Multiple PSCs with 75-100% control ---
    high_ctrl = [p for p in pscs
                 if "75-to-100-percent" in str(p.get("natures_of_control", []))]
    if len(high_ctrl) > 1:
        flags.append({"flag": f"{len(high_ctrl)} PSCs each claiming 75–100% control — unusual structure",
                      "severity": "High", "category": "Ownership Structure"})
        score += 15

    # --- Company age ---
    inc = profile.get("date_of_creation", "")
    if inc:
        try:
            age = (date.today() - datetime.strptime(inc, "%Y-%m-%d").date()).days
            if age < 180:
                flags.append({"flag": f"Newly incorporated — {age} days old",
                              "severity": "Medium", "category": "Company Age"})
                score += 10
        except Exception:
            pass

    # --- Accounts overdue ---
    if profile.get("accounts", {}).get("overdue", False):
        flags.append({"flag": "Annual accounts overdue",
                      "severity": "Medium", "category": "Filing Compliance"})
        score += 10

    # --- No filing history ---
    if not filings:
        flags.append({"flag": "No filing history found",
                      "severity": "Medium", "category": "Filing Compliance"})
        score += 10
    else:
        try:
            last = datetime.strptime(filings[0].get("date",""), "%Y-%m-%d").date()
            months = (date.today() - last).days / 30
            if months > 18:
                flags.append({"flag": f"No filings in {int(months)} months",
                              "severity": "Medium", "category": "Filing Compliance"})
                score += 10
        except Exception:
            pass

    # --- Panama Papers ---
    names = ([o.get("name","") for o in officers] +
             [p.get("name","") for p in pscs])
    hits  = [n for n in names if panama_hit(n)]
    if hits:
        flags.append({"flag": f"⚠️ {len(hits)} name(s) matched in Panama Papers leak: {', '.join(hits)}",
                      "severity": "Critical", "category": "Panama Papers"})
        score += 30

    score = min(score, 100)
    tier  = ("CRITICAL" if score >= 85 else
             "HIGH"     if score >= 60 else
             "MEDIUM"   if score >= 35 else "LOW")
    color = ("danger"  if tier in ("CRITICAL","HIGH") else
             "warning" if tier == "MEDIUM" else "success")

    return {
        "score":      score,
        "tier":       tier,
        "color":      color,
        "flags":      flags,
        "panama_hits": hits
    }

# ============================================================
# 5. Network graph
# ============================================================
def build_network(company_name, officers, pscs, panama_hits_set):
    G = nx.Graph()
    G.add_node(company_name, node_type="company",
               panama=company_name.lower() in panama_hits_set)

    for o in officers:
        name = o.get("name", "Unknown")
        G.add_node(name, node_type="officer", role=o.get("officer_role",""),
                   panama=panama_hit(name))
        G.add_edge(company_name, name)

    for p in pscs:
        name = p.get("name", "Unknown PSC")
        noc  = ", ".join(p.get("natures_of_control",[]))
        G.add_node(name, node_type="psc", noc=noc, panama=panama_hit(name))
        G.add_edge(company_name, name)

    if len(G.nodes) < 2:
        return go.Figure(layout=go.Layout(
            title="No network data available",
            paper_bgcolor="white"
        ))

    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for e in G.edges():
        edge_x += [pos[e[0]][0], pos[e[1]][0], None]
        edge_y += [pos[e[0]][1], pos[e[1]][1], None]

    def nc(n):
        if G.nodes[n].get("panama"): return "#e60000"
        t = G.nodes[n].get("node_type","")
        return "#1f4e79" if t=="company" else "#2471a3" if t=="officer" else "#1abc9c"

    hover = [
        f"<b>{n}</b><br>Type: {G.nodes[n].get('node_type','')}"
        f"{'<br>Role: ' + G.nodes[n].get('role','') if G.nodes[n].get('role') else ''}"
        f"{'<br>Control: ' + G.nodes[n].get('noc','') if G.nodes[n].get('noc') else ''}"
        f"{'<br>⚠️ PANAMA PAPERS HIT' if G.nodes[n].get('panama') else ''}"
        for n in G.nodes()
    ]

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        text=list(G.nodes()),
        textposition="bottom center",
        hovertext=hover, hoverinfo="text",
        marker=dict(
            color=[nc(n) for n in G.nodes()],
            size=[24 if G.nodes[n].get("node_type")=="company" else
                  20 if G.nodes[n].get("panama") else 16 for n in G.nodes()],
            symbol=["square" if G.nodes[n].get("node_type")=="company" else
                    "star"   if G.nodes[n].get("panama") else
                    "circle" for n in G.nodes()],
            line=dict(width=2, color="white")
        )
    )
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=1.5, color="#ccc"), hoverinfo="none")

    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Ownership & Control Network — {company_name}",
            showlegend=False, hovermode="closest",
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white", plot_bgcolor="#f9f9f9",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

# ============================================================
# 6. Portfolio analytics charts (meaningful only)
# ============================================================
def build_analytics_charts():
    """
    Returns 3 meaningful analyst charts:
    1. Risk tier distribution (deduplicated)
    2. Risk flag categories breakdown
    3. Filing compliance status
    """
    # Risk distribution — deduplicated
    risk_data = db_query("""
        SELECT DISTINCT ON (company_number) risk_tier, risk_score
        FROM kyc_status
        ORDER BY company_number, id DESC
    """)

    if not risk_data:
        empty = go.Figure()
        return empty, empty, empty

    df = pd.DataFrame(risk_data)

    # Chart 1 — Risk tier donut
    counts = df["risk_tier"].value_counts().reset_index()
    counts.columns = ["Risk Tier", "Count"]
    fig1 = px.pie(
        counts, names="Risk Tier", values="Count",
        title="Portfolio Risk Tier Distribution",
        color="Risk Tier",
        color_discrete_map={
            "CRITICAL": "#922b21", "HIGH": "#e74c3c",
            "MEDIUM": "#f39c12",   "LOW": "#27ae60"
        },
        hole=0.45
    )
    fig1.update_traces(textinfo="label+percent+value")
    fig1.update_layout(paper_bgcolor="white")

    # Chart 2 — Risk score histogram
    fig2 = px.histogram(
        df, x="risk_score", nbins=10,
        title="Risk Score Distribution Across Portfolio",
        labels={"risk_score": "Risk Score (0–100)", "count": "Number of Companies"},
        color_discrete_sequence=["#2e86c1"]
    )
    fig2.add_vline(x=35, line_dash="dash", line_color="#f39c12",
                   annotation_text="Medium threshold")
    fig2.add_vline(x=60, line_dash="dash", line_color="#e74c3c",
                   annotation_text="High threshold")
    fig2.update_layout(paper_bgcolor="white", plot_bgcolor="#f9f9f9")

    # Chart 3 — Accounts overdue vs on track
    compliance = db_query("""
        SELECT
            SUM(CASE WHEN accounts_overdue THEN 1 ELSE 0 END) AS overdue,
            SUM(CASE WHEN NOT accounts_overdue THEN 1 ELSE 0 END) AS on_track
        FROM (
            SELECT DISTINCT ON (company_number) company_number, accounts_overdue
            FROM companies ORDER BY company_number
        ) deduped
    """)
    if compliance:
        c = compliance[0]
        fig3 = go.Figure(go.Bar(
            x=["Accounts On Track", "Accounts Overdue"],
            y=[c.get("on_track", 0), c.get("overdue", 0)],
            marker_color=["#27ae60", "#e74c3c"],
            text=[c.get("on_track", 0), c.get("overdue", 0)],
            textposition="auto"
        ))
        fig3.update_layout(
            title="Filing Compliance — Accounts Status",
            paper_bgcolor="white", plot_bgcolor="#f9f9f9",
            yaxis_title="Number of Companies"
        )
    else:
        fig3 = go.Figure()

    return fig1, fig2, fig3

# ============================================================
# 7. App Layout
# ============================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "UBO / KYC Due Diligence Dashboard"

CARD = {
    "borderRadius": "8px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
    "marginBottom": "16px"
}

SEV_COLOR = {
    "Critical": "danger",
    "High":     "warning",
    "Medium":   "info",
    "Low":      "success"
}

app.layout = dbc.Container([

    # ---- Header ----
    dbc.Row(dbc.Col(html.Div([
        html.H2("🔍 UBO / KYC Due Diligence Dashboard", className="text-white mb-1"),
        html.Small(
            "Companies House Live API  ·  Panama Papers Screening  "
            "·  FATF Jurisdiction Risk  ·  PostgreSQL",
            className="text-white-50"
        )
    ], style={
        "background": "linear-gradient(135deg,#1f4e79,#2e86c1)",
        "padding": "20px 30px", "borderRadius": "10px", "marginBottom": "20px"
    }))),

    # ---- Tabs ----
    dbc.Tabs([

        # ======================================================
        # TAB 1 — Due Diligence (main tab)
        # ======================================================
        dbc.Tab(label="🔍 Due Diligence", tab_id="tab-dd", children=[
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id="company-select",
                    placeholder="Type or select a company to begin due diligence…",
                    clearable=True,
                    style={"fontSize": "0.95rem"}
                ), width=9),
                dbc.Col(dbc.Button(
                    "▶ Review", id="load-btn",
                    color="primary", className="w-100"
                ), width=3),
            ], className="mb-3"),

            dcc.Loading(type="circle", color="#1f4e79", children=[

                # Risk summary banner — shown after selection
                html.Div(id="risk-banner"),

                # Risk flags table
                html.Div(id="risk-flags-panel"),

                # Two column — profile + PSC
                dbc.Row([
                    dbc.Col(html.Div(id="profile-panel"), width=6),
                    dbc.Col(html.Div(id="psc-panel"),     width=6),
                ]),

                # Officers
                html.Div(id="officers-panel"),

                # Panama Papers
                html.Div(id="panama-panel"),

                # Filing history
                html.Div(id="filings-panel"),

                # Network graph
                html.Div(id="network-panel"),
            ])
        ]),

        # ======================================================
        # TAB 2 — Portfolio Overview
        # ======================================================
        dbc.Tab(label="📋 Portfolio Overview", tab_id="tab-portfolio", children=[
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Input(
                    id="portfolio-search",
                    placeholder="Filter by name or company number…",
                    debounce=True
                ), width=4),
                dbc.Col(dcc.Dropdown(
                    id="tier-filter",
                    options=[{"label": t, "value": t}
                             for t in ["ALL","LOW","MEDIUM","HIGH","CRITICAL"]],
                    value="ALL", clearable=False
                ), width=2),
                dbc.Col(dcc.Dropdown(
                    id="panama-filter",
                    options=[{"label": "All Companies",        "value": "ALL"},
                             {"label": "⚠️ Panama Hits Only", "value": "HIT"}],
                    value="ALL", clearable=False
                ), width=3),
                dbc.Col(dbc.Button(
                    "🔄 Refresh", id="refresh-btn", color="secondary"
                ), width=1),
            ], className="mb-3"),
            html.Div(id="portfolio-table"),
            dcc.Store(id="customer-list-store"),
            dcc.Interval(id="kpi-interval", interval=60*1000, n_intervals=0),
        ]),

        # ======================================================
        # TAB 3 — Portfolio Analytics
        # ======================================================
        dbc.Tab(label="📊 Portfolio Analytics", tab_id="tab-analytics", children=[
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Graph(id="risk-dist-chart"),   width=6),
                dbc.Col(dcc.Graph(id="risk-score-chart"),  width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="compliance-chart"),  width=6),
                dbc.Col(html.Div(id="fatf-summary-panel"), width=6),
            ]),
        ]),

    ], id="main-tabs", active_tab="tab-dd"),

], fluid=True, style={
    "padding": "24px",
    "backgroundColor": "#f0f4f8",
    "minHeight": "100vh"
})

# ============================================================
# 8. Populate company dropdown on load
# ============================================================
@app.callback(
    Output("company-select", "options"),
    Input("kpi-interval", "n_intervals")
)
def populate_dropdown(_):
    return load_company_options()

# ============================================================
# 9. Portfolio table
# ============================================================
@app.callback(
    Output("customer-list-store", "data"),
    Output("portfolio-table",     "children"),
    Input("kpi-interval",    "n_intervals"),
    Input("refresh-btn",     "n_clicks"),
    Input("portfolio-search","value"),
    Input("tier-filter",     "value"),
    Input("panama-filter",   "value"),
)
def update_portfolio(_, __, search, tier, panama_f):
    rows = db_query("""
        SELECT DISTINCT ON (c.company_number)
               c.company_number, c.company_name, c.company_status,
               k.risk_tier, k.risk_score, k.panama_papers_hit, k.status AS kyc_status
        FROM companies c
        LEFT JOIN kyc_status k ON c.company_number = k.company_number
        ORDER BY c.company_number, k.id DESC
    """)

    if not rows:
        return [], dbc.Alert("No companies in database. Run seed_database.py first.", color="warning")

    df = pd.DataFrame(rows)

    if search:
        mask = (df["company_name"].str.contains(search, case=False, na=False) |
                df["company_number"].str.contains(search, case=False, na=False))
        df = df[mask]

    if tier and tier != "ALL":
        df = df[df["risk_tier"] == tier]

    if panama_f == "HIT":
        df = df[df["panama_papers_hit"] == True]

    df["risk_score"]        = df["risk_score"].fillna(0).astype(int)
    df["panama_papers_hit"] = df["panama_papers_hit"].apply(
        lambda x: "⚠️ YES" if x else "✅ No"
    )

    display = df.rename(columns={
        "company_number":   "Company No.",
        "company_name":     "Company Name",
        "company_status":   "CH Status",
        "risk_tier":        "Risk Tier",
        "risk_score":       "Score",
        "panama_papers_hit":"Panama",
        "kyc_status":       "KYC Status"
    })[["Company No.","Company Name","CH Status","Risk Tier","Score","Panama","KYC Status"]]

    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in display.columns],
        data=display.to_dict("records"),
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1f4e79",
            "color": "white",
            "fontWeight": "bold"
        },
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "0.85rem"},
        style_data_conditional=[
            {"if": {"filter_query": '{Risk Tier} = "HIGH"'},
             "backgroundColor": "#ffe0e0", "color": "#c0392b"},
            {"if": {"filter_query": '{Risk Tier} = "CRITICAL"'},
             "backgroundColor": "#ffcccc", "color": "#922b21", "fontWeight": "bold"},
            {"if": {"filter_query": '{Panama} = "⚠️ YES"', "column_id": "Panama"},
             "color": "red", "fontWeight": "bold"},
        ]
    )
    return rows, table

# ============================================================
# 10. Due Diligence — main callback
# ============================================================
@app.callback(
    Output("risk-banner",      "children"),
    Output("risk-flags-panel", "children"),
    Output("profile-panel",    "children"),
    Output("psc-panel",        "children"),
    Output("officers-panel",   "children"),
    Output("panama-panel",     "children"),
    Output("filings-panel",    "children"),
    Output("network-panel",    "children"),
    Input("load-btn",          "n_clicks"),
    State("company-select",    "value"),
    prevent_initial_call=True
)
def run_due_diligence(_, company_number):
    if not company_number:
        empty = dbc.Alert("Please select a company first.", color="info")
        return empty, "", "", "", "", "", "", ""

    # Fetch live data
    profile  = ch_get(f"/company/{company_number}") or {}
    officers = get_officers(company_number)
    pscs     = get_pscs(company_number)
    filings  = get_filings(company_number)

    company_name    = profile.get("company_name", company_number)
    risk            = compute_risk(profile, officers, pscs, filings)
    panama_hits_set = {h.lower() for h in risk["panama_hits"]}

    # ---- RISK BANNER ----
    tier_colors = {
        "CRITICAL": "#922b21", "HIGH": "#c0392b",
        "MEDIUM": "#d35400",   "LOW": "#1e8449"
    }
    banner = dbc.Card(
        dbc.CardBody(dbc.Row([
            dbc.Col([
                html.H4(company_name, className="mb-1 text-white"),
                html.Small(f"Company No: {company_number}  ·  "
                           f"Status: {profile.get('company_status','N/A').title()}",
                           className="text-white-50")
            ], width=7),
            dbc.Col([
                html.H2(f"{risk['score']} / 100",
                        className="text-white mb-0 text-end"),
                html.Div(
                    dbc.Badge(f"⚠️ {risk['tier']} RISK",
                              style={"fontSize": "1rem",
                                     "backgroundColor": tier_colors.get(risk["tier"],"#555")}),
                    className="text-end"
                )
            ], width=5),
        ])),
        style={
            "background": f"linear-gradient(135deg,{tier_colors.get(risk['tier'],'#555')},#2c3e50)",
            "borderRadius": "10px", "marginBottom": "16px"
        }
    )

    # ---- RISK FLAGS TABLE ----
    if risk["flags"]:
        flag_rows = []
        for f in sorted(risk["flags"],
                        key=lambda x: ["Critical","High","Medium","Low"].index(x["severity"])
                        if x["severity"] in ["Critical","High","Medium","Low"] else 99):
            flag_rows.append({
                "Category":  f["category"],
                "Risk Flag": f["flag"],
                "Severity":  f["severity"]
            })

        flags_panel = dbc.Card([
            dbc.CardHeader(html.H6("🚩 Risk Flags Identified", className="mb-0")),
            dbc.CardBody([
                dbc.Progress(
                    value=risk["score"], color=risk["color"],
                    className="mb-3", style={"height": "10px"}
                ),
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in ["Category","Risk Flag","Severity"]],
                    data=flag_rows,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#1f4e79",
                        "color": "white", "fontWeight": "bold"
                    },
                    style_cell={"textAlign": "left", "padding": "8px", "fontSize": "0.85rem"},
                    style_data_conditional=[
                        {"if": {"filter_query": '{Severity} = "Critical"', "column_id": "Severity"},
                         "backgroundColor": "#ffcccc", "color": "#922b21", "fontWeight": "bold"},
                        {"if": {"filter_query": '{Severity} = "High"', "column_id": "Severity"},
                         "backgroundColor": "#ffe0e0", "color": "#c0392b"},
                        {"if": {"filter_query": '{Severity} = "Medium"', "column_id": "Severity"},
                         "backgroundColor": "#fef9e7", "color": "#d35400"},
                    ]
                )
            ])
        ], style=CARD)
    else:
        flags_panel = dbc.Card([
            dbc.CardHeader(html.H6("✅ Risk Assessment", className="mb-0")),
            dbc.CardBody(dbc.Alert("No risk flags identified for this company.", color="success"))
        ], style=CARD)

    # ---- COMPANY PROFILE ----
    addr = profile.get("registered_office_address", {})
    addr_str = ", ".join(filter(None, [
        addr.get("address_line_1"), addr.get("address_line_2"),
        addr.get("locality"), addr.get("postal_code"), addr.get("country")
    ]))
    status     = profile.get("company_status","N/A").replace("_"," ").title()
    status_col = "success" if status.lower() == "active" else "danger"

    jx_label, jx_sev = jurisdiction_risk(addr.get("country",""))
    jx_badge = dbc.Badge(
        f"⚠️ {jx_label}", color="danger" if jx_sev == "Critical" else "warning",
        className="ms-2"
    ) if jx_label else ""

    profile_panel = dbc.Card([
        dbc.CardHeader(html.H6("🏢 Company Profile", className="mb-0")),
        dbc.CardBody(html.Table([html.Tbody([
            html.Tr([html.Th("Company Name"),   html.Td(company_name)]),
            html.Tr([html.Th("Company Number"), html.Td(company_number)]),
            html.Tr([html.Th("Status"),         html.Td(dbc.Badge(status, color=status_col))]),
            html.Tr([html.Th("Incorporated"),   html.Td(profile.get("date_of_creation","N/A"))]),
            html.Tr([html.Th("Company Type"),   html.Td(profile.get("type","N/A"))]),
            html.Tr([html.Th("Jurisdiction"),   html.Td(profile.get("jurisdiction","N/A"))]),
            html.Tr([html.Th("SIC Codes"),      html.Td(
                ", ".join(profile.get("sic_codes",[])) or "N/A"
            )]),
            html.Tr([html.Th("Registered Address"), html.Td([addr_str or "N/A", jx_badge])]),
            html.Tr([html.Th("Accounts Due"),   html.Td(
                profile.get("accounts",{}).get("next_due","N/A")
            )]),
            html.Tr([html.Th("Accounts Overdue"), html.Td(
                dbc.Badge("YES", color="danger")
                if profile.get("accounts",{}).get("overdue") else
                dbc.Badge("No", color="success")
            )]),
        ])], className="table table-sm table-striped mb-0"))
    ], style=CARD)

    # ---- PSC PANEL ----
    if pscs:
        psc_rows = []
        for p in pscs:
            name    = p.get("name","N/A")
            noc     = ", ".join(p.get("natures_of_control",[]))
            country = p.get("country_of_residence", p.get("nationality","N/A"))
            dob     = p.get("date_of_birth",{})
            dob_str = f"{dob.get('month','')}/{dob.get('year','')}" if dob else "N/A"
            hit     = panama_hit(name)
            jl, js  = jurisdiction_risk(country)
            psc_rows.append(html.Tr([
                html.Td([
                    name,
                    dbc.Badge("⚠️ Panama", color="danger", className="ms-1") if hit else "",
                    dbc.Badge(f"⚠️ {jl}", color="warning", className="ms-1") if jl else ""
                ]),
                html.Td(noc, style={"fontSize":"0.8rem"}),
                html.Td(country),
                html.Td(dob_str),
            ], style={"backgroundColor":"#fff0f0"} if (hit or jl) else {}))

        psc_panel = dbc.Card([
            dbc.CardHeader(html.H6(f"👤 Persons with Significant Control ({len(pscs)})",
                                   className="mb-0")),
            dbc.CardBody(html.Table([
                html.Thead(html.Tr([
                    html.Th("Name"), html.Th("Nature of Control"),
                    html.Th("Country"), html.Th("DOB")
                ])),
                html.Tbody(psc_rows)
            ], className="table table-sm table-striped mb-0"))
        ], style=CARD)
    else:
        psc_panel = dbc.Card([
            dbc.CardHeader(html.H6("👤 Persons with Significant Control", className="mb-0")),
            dbc.CardBody(dbc.Alert(
                "⚠️ No PSC records registered — ownership structure is opaque. "
                "This is a significant red flag requiring further investigation.",
                color="danger"
            ))
        ], style=CARD)

    # ---- OFFICERS PANEL ----
    if officers:
        officer_data = []
        for o in officers:
            name     = o.get("name","N/A")
            nat      = o.get("nationality","N/A")
            cor      = o.get("country_of_residence","N/A")
            hit      = panama_hit(name)
            jl_nat, js_nat = jurisdiction_risk(nat)
            jl_cor, js_cor = jurisdiction_risk(cor)
            jx_flag  = jl_nat or jl_cor or ""
            officer_data.append({
                "Name":               name,
                "Role":               o.get("officer_role","N/A").replace("-"," ").title(),
                "Appointed":          o.get("appointed_on","N/A"),
                "Status":             "Resigned" if o.get("resigned_on") else "Active",
                "Nationality":        nat,
                "Country of Residence": cor,
                "Jurisdiction Flag":  f"⚠️ {jx_flag}" if jx_flag else "✅ Clear",
                "Panama Hit":         "⚠️ YES" if hit else "✅ No"
            })

        officers_panel = dbc.Card([
            dbc.CardHeader(html.H6(f"🧑‍💼 Officers & Directors ({len(officers)})",
                                   className="mb-0")),
            dbc.CardBody(dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in officer_data[0].keys()],
                data=officer_data,
                page_size=10,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX":"auto"},
                style_header={
                    "backgroundColor":"#1f4e79",
                    "color":"white","fontWeight":"bold"
                },
                style_cell={"textAlign":"left","padding":"8px","fontSize":"0.85rem"},
                style_data_conditional=[
                    {"if": {"filter_query": '{Panama Hit} = "⚠️ YES"'},
                     "backgroundColor": "#ffe0e0", "color": "red", "fontWeight": "bold"},
                    {"if": {"filter_query": 'contains({Jurisdiction Flag}, "⚠️")'},
                     "backgroundColor": "#fff3cd"},
                ]
            ))
        ], style=CARD)
    else:
        officers_panel = dbc.Card([
            dbc.CardHeader(html.H6("🧑‍💼 Officers & Directors", className="mb-0")),
            dbc.CardBody(html.P("No officer records found."))
        ], style=CARD)

    # ---- PANAMA PANEL ----
    if risk["panama_hits"]:
        panama_panel = dbc.Card([
            dbc.CardHeader(html.H6("🚨 Panama Papers — Matched Names",
                                   style={"color":"red"}, className="mb-0")),
            dbc.CardBody([
                dbc.Alert(
                    f"⚠️ {len(risk['panama_hits'])} officer/PSC name(s) appear in the "
                    f"Panama Papers dataset. Enhanced Due Diligence is required.",
                    color="danger"
                ),
                html.Ul([html.Li(h) for h in risk["panama_hits"]])
            ])
        ], style=CARD, className="border-danger")
    else:
        panama_panel = dbc.Card([
            dbc.CardHeader(html.H6("Panama Papers Screening", className="mb-0")),
            dbc.CardBody(dbc.Alert(
                "✅ No officer or PSC names matched in the Panama Papers dataset.",
                color="success"
            ))
        ], style=CARD)

    # ---- FILINGS PANEL ----
    if filings:
        filing_data = [{
            "Date":        f.get("date","N/A"),
            "Type":        f.get("type","N/A"),
            "Description": f.get("description","N/A"),
            "Category":    f.get("category","N/A")
        } for f in filings]

        filings_panel = dbc.Card([
            dbc.CardHeader(html.H6("📁 Filing History (Last 15)", className="mb-0")),
            dbc.CardBody(dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in filing_data[0].keys()],
                data=filing_data,
                page_size=10,
                sort_action="native",
                style_table={"overflowX":"auto"},
                style_header={
                    "backgroundColor":"#1f4e79",
                    "color":"white","fontWeight":"bold"
                },
                style_cell={"textAlign":"left","padding":"8px","fontSize":"0.85rem"}
            ))
        ], style=CARD)
    else:
        filings_panel = dbc.Card([
            dbc.CardHeader(html.H6("📁 Filing History", className="mb-0")),
            dbc.CardBody(dbc.Alert("⚠️ No filing history found.", color="warning"))
        ], style=CARD)

    # ---- NETWORK GRAPH ----
    fig = build_network(company_name, officers, pscs, panama_hits_set)
    network_panel = dbc.Card([
        dbc.CardHeader(html.H6("🕸️ Ownership & Control Network Graph", className="mb-0")),
        dbc.CardBody(dbc.Row([
            dbc.Col(dcc.Graph(figure=fig, style={"height":"600px"}), width=10),
            dbc.Col([
                html.H6("Legend", className="mt-3 fw-bold"),
                html.Ul([
                    html.Li("🟦 Company (square)",    style={"color":"#1f4e79"}),
                    html.Li("🔵 Officer (circle)",     style={"color":"#2471a3"}),
                    html.Li("🟢 PSC / UBO (circle)",   style={"color":"#1abc9c"}),
                    html.Li("⭐ Panama Hit (star/red)", style={"color":"#e60000",
                                                               "fontWeight":"bold"}),
                ], style={"fontSize":"0.85rem","lineHeight":"2.2rem",
                           "paddingLeft":"16px","listStyle":"none"})
            ], width=2)
        ]))
    ], style=CARD)

    return (banner, flags_panel, profile_panel, psc_panel,
            officers_panel, panama_panel, filings_panel, network_panel)

# ============================================================
# 11. Portfolio Analytics charts
# ============================================================
@app.callback(
    Output("risk-dist-chart",  "figure"),
    Output("risk-score-chart", "figure"),
    Output("compliance-chart", "figure"),
    Output("fatf-summary-panel","children"),
    Input("kpi-interval", "n_intervals"),
    Input("refresh-btn",  "n_clicks")
)
def update_analytics(_, __):
    fig1, fig2, fig3 = build_analytics_charts()

    # FATF summary panel — counts companies in high-risk jurisdictions
    fatf_data = db_query("""
        SELECT DISTINCT ON (company_number) company_number, company_name, country
        FROM companies ORDER BY company_number
    """)

    fatf_rows = []
    for r in fatf_data:
        jl, js = jurisdiction_risk(r.get("country",""))
        if jl:
            fatf_rows.append({
                "Company":    r["company_name"],
                "Country":    r["country"],
                "Risk Label": jl,
                "Severity":   js
            })

    if fatf_rows:
        fatf_panel = dbc.Card([
            dbc.CardHeader(html.H6(
                f"🌍 FATF / Jurisdiction Risk — {len(fatf_rows)} Company(ies) Flagged",
                className="mb-0", style={"color":"#c0392b"}
            )),
            dbc.CardBody(dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in fatf_rows[0].keys()],
                data=fatf_rows,
                style_table={"overflowX":"auto"},
                style_header={
                    "backgroundColor":"#1f4e79",
                    "color":"white","fontWeight":"bold"
                },
                style_cell={"textAlign":"left","padding":"8px","fontSize":"0.85rem"},
                style_data_conditional=[
                    {"if": {"filter_query": '{Severity} = "Critical"'},
                     "backgroundColor":"#ffcccc","color":"#922b21","fontWeight":"bold"},
                    {"if": {"filter_query": '{Severity} = "High"'},
                     "backgroundColor":"#ffe0e0","color":"#c0392b"},
                ]
            ))
        ], style=CARD)
    else:
        fatf_panel = dbc.Card([
            dbc.CardHeader(html.H6("🌍 FATF / Jurisdiction Risk", className="mb-0")),
            dbc.CardBody(dbc.Alert(
                "✅ No companies registered in FATF grey/black list jurisdictions.",
                color="success"
            ))
        ], style=CARD)

    return fig1, fig2, fig3, fatf_panel

# ============================================================
# 12. Run
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
