# ============================================================
# IMPORTS
# ============================================================
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================
# 1. SPARK SESSION
# ============================================================
spark = SparkSession.builder \
    .appName("GlobalRiskIntelligenceDashboard") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")   # Suppress INFO noise in terminal

# ============================================================
# 2. LOAD DATA INTO SPARK (stays in cluster memory)
#    All DataFrames are cached and registered as Temp Views
#    so Spark SQL can optimise joins across them.
# ============================================================
def load_and_register(hdfs_path: str, view_name: str):
    """
    Reads a CSV from HDFS into a Spark DataFrame, caches it in
    cluster memory, and registers it as a Spark SQL Temp View.
    Nothing is collected to the driver here.
    """
    df = (
        spark.read
             .option("header", "true")
             .option("inferSchema", "true")
             .option("sep", ",")
             .option("multiLine", "true")
             .option("escape", '"')
             .csv(hdfs_path)
             .withColumn(               # normalise company_id everywhere
                 "company_id",
                 F.lower(F.trim(F.col("company_id")))
             )
             .cache()
    )
    df.createOrReplaceTempView(view_name)
    return df

companies_sdf    = load_and_register("hdfs:///data/synthetic_companies_house.csv",        "companies")  ##DEFINED_DATABASE_DIRECTORY##
psc_sdf          = load_and_register("hdfs:///data/synthetic_psc_register.csv",           "psc")  ##DEFINED_DATABASE_DIRECTORY##
transactions_sdf = load_and_register("hdfs:///data/synthetic_financial_transactions.csv", "transactions")  ##DEFINED_DATABASE_DIRECTORY##
risk_scores_sdf  = load_and_register("hdfs:///data/synthetic_risk_scores.csv",            "risk_scores")  ##DEFINED_DATABASE_DIRECTORY##
social_media_sdf = load_and_register("hdfs:///data/synthetic_social_media_data.csv",      "social_media") ##DEFINED_DATABASE_DIRECTORY##

# ============================================================
# 3. SPARK UDF — VADER SENTIMENT
# ============================================================
_analyzer = SentimentIntensityAnalyzer()

def _vader_score(text: str) -> float:
    """Returns compound VADER score normalised to [0, 1]."""
    if not isinstance(text, str) or not text.strip():
        return 0.5
    raw = _analyzer.polarity_scores(text)["compound"]   # range -1 to +1
    return round((raw + 1) / 2, 4)                      # normalise to 0–1

vader_udf = F.udf(_vader_score, FloatType())

# Pre-compute sentiment for all social media rows on the cluster
# and cache the enriched view — this runs once at startup.
social_media_sdf = (
    social_media_sdf
    .withColumn("sentiment_score", vader_udf(F.col("post_content")))
    .cache()
)
social_media_sdf.createOrReplaceTempView("social_media")   # refresh view

# ============================================================
# 4. LIGHTWEIGHT DRIVER-SIDE LOOKUPS
#    We collect ONLY the company dropdown list (IDs + names).
#    This is a tiny result set — safe to bring to the driver.
# ============================================================
company_lookup_df = spark.sql(
    "SELECT company_id, company_name FROM companies ORDER BY company_name"
).toPandas()

company_options = [
    {"label": row["company_name"], "value": row["company_id"]}
    for _, row in company_lookup_df.iterrows()
]

# ============================================================
# 5. DASH APP LAYOUT
# ============================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Risk Intelligence Dashboard"

app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("🔍 Global Risk Intelligence Dashboard", className="fw-bold fs-5")
        ]),
        color="dark", dark=True, className="mb-4"
    ),

    dbc.Container([

        # ── Global company selector ──────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Company / Vendor:", className="fw-semibold"),
                dcc.Dropdown(
                    id="company-dropdown",
                    options=company_options,
                    value=company_options[0]["value"] if company_options else None,
                    clearable=False
                )
            ], width=6)
        ], className="mb-4"),

        # ── Tabs ────────────────────────────────────────────
        dcc.Tabs(id="main-tabs", children=[

            # ─────────── TAB 1: Company Structure ───────────
            dcc.Tab(label="Company Structure", children=[
                dbc.Row([
                    dbc.Col(dcc.Loading(html.Div(id="company-info-content")),  width=6),
                    dbc.Col(dcc.Loading(html.Div(id="psc-info-content")),      width=6),
                ], className="mt-3")
            ]),

            # ─────────── TAB 2: Financial Transactions ──────
            dcc.Tab(label="Financial Transactions", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Filters", className="mb-3"),
                    dbc.Row([
                        dbc.Col(dcc.Input(
                            id="filter-sender",
                            type="text",
                            placeholder="Sender account contains…",
                            debounce=True,
                            className="form-control"
                        ), width=4),
                        dbc.Col(dcc.Dropdown(
                            id="filter-txn-type",
                            placeholder="Transaction type",
                            multi=True
                        ), width=4),
                        dbc.Col(dcc.Dropdown(
                            id="filter-risk",
                            options=[
                                {"label": "Low  (≤ 0.50)",       "value": "low"},
                                {"label": "Medium (0.51 – 0.70)", "value": "medium"},
                                {"label": "High  (> 0.70)",       "value": "high"},
                            ],
                            placeholder="Financial risk band",
                            clearable=True
                        ), width=4),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dcc.DatePickerRange(
                            id="txn-date-range",
                            display_format="YYYY-MM-DD",
                            start_date_placeholder_text="Start date",
                            end_date_placeholder_text="End date"
                        ), width=8),
                        dbc.Col(dbc.Button(
                            "Apply Filter", id="filter-btn",
                            color="primary", n_clicks=0
                        ), width=2),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col(dbc.Button("⬇ Download CSV",  id="download-csv-btn",  color="secondary", size="sm"), width="auto"),
                        dbc.Col(dbc.Button("⬇ Download JSON", id="download-json-btn", color="secondary", size="sm"), width="auto"),
                    ], className="mb-3"),
                    dcc.Download(id="download-csv"),
                    dcc.Download(id="download-json"),
                    # Hidden store — keeps the current filtered data for downloads
                    dcc.Store(id="filtered-txn-store"),
                    dcc.Loading(html.Div(id="financial-transactions-content"))
                ]), className="mt-3")
            ]),

            # ─────────── TAB 3: Risk Dashboard ──────────────
            dcc.Tab(label="Risk Dashboard", children=[
                dbc.Row([
                    dbc.Col([
                        html.H5(id="risk-tab-company-header", className="mt-3 mb-2"),
                        dcc.Loading(html.Div(id="high-risk-summary")),
                        dcc.Loading(dcc.Graph(id="risk-trend-graph")),
                    ], width=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H5("Social Media Sentiment", className="mt-3"),
                        dcc.Loading(html.Div(id="social-media-posts-content")),
                        dcc.Loading(html.Div(id="fraud-flags")),
                    ], width=7),
                    dbc.Col([
                        dcc.Loading(dcc.Graph(id="sentiment-gauge")),
                    ], width=5),
                ])
            ]),
        ])
    ], fluid=True)
])

# ============================================================
# 6. CALLBACKS
# ============================================================

# ── 6-A: Populate transaction-type dropdown from Spark ──────
@app.callback(
    Output("filter-txn-type", "options"),
    Input("company-dropdown", "value")
)
def populate_txn_types(company_id):
    """
    Fetches distinct transaction types for the selected company.
    Only the distinct values are collected — still tiny.
    """
    if not company_id:
        return []
    types_df = spark.sql(f"""
        SELECT DISTINCT transaction_type
        FROM transactions
        WHERE company_id = '{company_id}'
        ORDER BY transaction_type
    """).toPandas()
    return [{"label": t, "value": t} for t in types_df["transaction_type"].dropna()]


# ── 6-B: Company Structure tab ──────────────────────────────
@app.callback(
    Output("company-info-content", "children"),
    Input("company-dropdown", "value")
)
def update_company_info(company_id):
    if not company_id:
        return html.Div("Select a company above.")

    df = spark.sql(f"""
        SELECT company_name, registration_number, incorporation_date,
               registered_address, nature_of_business, status
        FROM companies
        WHERE company_id = '{company_id}'
        LIMIT 1
    """).toPandas()

    if df.empty:
        return html.Div("⚠️ No company record found.", className="text-warning")

    c = df.iloc[0]
    return html.Div([
        html.H5(f"{c['company_name']}", className="mb-3"),
        dbc.ListGroup([
            dbc.ListGroupItem([html.Strong("Reg. Number: "),     str(c["registration_number"])]),
            dbc.ListGroupItem([html.Strong("Incorporated: "),    str(c["incorporation_date"])]),
            dbc.ListGroupItem([html.Strong("Address: "),         str(c["registered_address"])]),
            dbc.ListGroupItem([html.Strong("Nature of Business: "), str(c["nature_of_business"])]),
            dbc.ListGroupItem([html.Strong("Status: "),
                dbc.Badge(str(c["status"]),
                    color="success" if str(c["status"]).lower() == "active" else "danger")
            ]),
        ])
    ])


@app.callback(
    Output("psc-info-content", "children"),
    Input("company-dropdown", "value")
)
def update_psc_info(company_id):
    if not company_id:
        return html.Div()

    df = spark.sql(f"""
        SELECT psc_id, psc_name, date_of_birth, nationality,
               psc_address, control_type
        FROM psc
        WHERE company_id = '{company_id}'
    """).toPandas()

    if df.empty:
        return html.Div("No Persons with Significant Control recorded.", className="text-muted")

    return html.Div([
        html.H5("Persons with Significant Control (UBO)", className="mb-3"),
        dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size="sm")
    ])


# ── 6-C: Financial Transactions — filter & store ────────────
@app.callback(
    [Output("financial-transactions-content", "children"),
     Output("filtered-txn-store", "data")],
    Input("filter-btn", "n_clicks"),
    State("company-dropdown", "value"),
    State("txn-date-range", "start_date"),
    State("txn-date-range", "end_date"),
    State("filter-txn-type", "value"),
    State("filter-sender", "value"),
    State("filter-risk", "value"),
    prevent_initial_call=False
)
def filter_transactions(n_clicks, company_id, start, end, types,
                        filter_sender, filter_risk):
    if not company_id:
        return html.Div("Select a company to view transactions."), None

    # Build Spark SQL predicate dynamically — all filtering on cluster
    predicates = [f"company_id = '{company_id}'"]

    if start:
        predicates.append(f"transaction_date >= '{start}'")
    if end:
        predicates.append(f"transaction_date <= '{end}'")
    if filter_sender:
        safe_sender = filter_sender.replace("'", "''")
        predicates.append(f"LOWER(sender_account) LIKE '%{safe_sender.lower()}%'")
    if types:
        quoted = ", ".join(f"'{t}'" for t in types)
        predicates.append(f"transaction_type IN ({quoted})")
    if filter_risk == "low":
        predicates.append("financial_risk <= 0.5")
    elif filter_risk == "medium":
        predicates.append("financial_risk > 0.5 AND financial_risk <= 0.7")
    elif filter_risk == "high":
        predicates.append("financial_risk > 0.7")

    where_clause = " AND ".join(predicates)

    query = f"""
        SELECT transaction_date, transaction_type, company_id,
               sender_account, recipient_account,
               transaction_amount, transaction_id, financial_risk
        FROM transactions
        WHERE {where_clause}
        ORDER BY transaction_date DESC
    """

    # Collect ONLY the filtered page — still bounded
    df = spark.sql(query).limit(5000).toPandas()

    if df.empty:
        return html.Div("No transactions match the selected filters.", className="text-muted"), None

    df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.strftime("%Y-%m-%d")

    style_cond = [
        {"if": {"filter_query": "{financial_risk} <= 0.5",  "column_id": "financial_risk"}, "color": "green"},
        {"if": {"filter_query": "{financial_risk} > 0.5 && {financial_risk} <= 0.7", "column_id": "financial_risk"}, "color": "darkorange"},
        {"if": {"filter_query": "{financial_risk} > 0.7",   "column_id": "financial_risk"}, "color": "red", "fontWeight": "bold"},
    ]

    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.to_dict("records"),
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_data_conditional=style_cond,
        style_header={"backgroundColor": "#343a40", "color": "white", "fontWeight": "bold"},
        export_format="csv"    # built-in export as safety net
    )

    summary = dbc.Alert(
        f"Showing {len(df):,} transactions  |  "
        f"Total amount: ${df['transaction_amount'].sum():,.2f}  |  "
        f"High-risk rows: {(df['financial_risk'] > 0.7).sum():,}",
        color="info", className="mb-2"
    )

    return html.Div([summary, table]), df.to_json(orient="records", date_format="iso")


# ── 6-D: Download CSV (uses filtered store) ─────────────────
@app.callback(
    Output("download-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("filtered-txn-store", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, json_data):
    if not json_data:
        return dash.no_update
    df = pd.read_json(json_data, orient="records")
    return dcc.send_data_frame(df.to_csv, "transactions_filtered.csv", index=False)


# ── 6-E: Download JSON (uses filtered store) ────────────────
@app.callback(
    Output("download-json", "data"),
    Input("download-json-btn", "n_clicks"),
    State("filtered-txn-store", "data"),
    prevent_initial_call=True
)
def download_json(n_clicks, json_data):
    if not json_data:
        return dash.no_update
    return dict(content=json_data, filename="transactions_filtered.json")


# ── 6-F: Risk Dashboard — trend chart + summary ─────────────
@app.callback(
    [Output("risk-tab-company-header", "children"),
     Output("risk-trend-graph", "figure"),
     Output("high-risk-summary", "children")],
    Input("company-dropdown", "value")
)
def update_risk_dashboard(company_id):
    if not company_id:
        return "No company selected.", go.Figure(), html.Div()

    # Company name for header
    name_df = spark.sql(f"""
        SELECT company_name FROM companies WHERE company_id = '{company_id}' LIMIT 1
    """).toPandas()

    company_name = name_df.iloc[0]["company_name"] if not name_df.empty else company_id

    # Risk score time series — small collect
    risk_df = spark.sql(f"""
        SELECT transaction_date, risk_score
        FROM risk_scores
        WHERE company_id = '{company_id}'
        ORDER BY transaction_date
    """).toPandas()

    if risk_df.empty:
        return (
            f"Risk Dashboard — {company_name}",
            go.Figure().update_layout(title="No risk data available"),
            html.Div("No risk score data found for this company.", className="text-muted")
        )

    risk_df["transaction_date"] = pd.to_datetime(risk_df["transaction_date"])

    fig = px.line(
        risk_df, x="transaction_date", y="risk_score",
        title=f"Risk Score Over Time — {company_name}",
        labels={"risk_score": "Risk Score", "transaction_date": "Date"}
    )
    fig.add_shape(
        type="line",
        x0=risk_df["transaction_date"].min(), x1=risk_df["transaction_date"].max(),
        y0=0.7, y1=0.7,
        line=dict(color="red", dash="dash")
    )
    fig.add_annotation(
        x=risk_df["transaction_date"].max(), y=0.72,
        text="High-risk threshold (0.70)", showarrow=False,
        font=dict(color="red", size=11)
    )

    # High-risk transaction summary — aggregated on Spark, tiny result
    summary_df = spark.sql(f"""
        SELECT
            COUNT(*)                                     AS high_risk_dates,
            COALESCE(SUM(t.transaction_amount), 0)       AS total_high_risk_value
        FROM risk_scores r
        LEFT JOIN transactions t
            ON r.company_id = t.company_id
           AND r.transaction_date = t.transaction_date
        WHERE r.company_id = '{company_id}'
          AND r.risk_score > 0.7
    """).toPandas()

    s = summary_df.iloc[0]
    summary_card = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("High-Risk Events"),
            dbc.CardBody(html.H4(f"{int(s['high_risk_dates'])}", className="text-danger"))
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("High-Risk Transaction Value"),
            dbc.CardBody(html.H4(f"${float(s['total_high_risk_value']):,.2f}", className="text-danger"))
        ]), width=4),
    ], className="mb-3")

    return f"Risk Dashboard — {company_name}", fig, summary_card


# ── 6-G: Social Media Sentiment & Fraud Flags ───────────────
@app.callback(
    [Output("social-media-posts-content", "children"),
     Output("sentiment-gauge", "figure"),
     Output("fraud-flags", "children")],
    Input("company-dropdown", "value")
)
def update_social_media(company_id):
    if not company_id:
        return html.Div(), go.Figure(), html.Div()

    # Average sentiment + post count — aggregated on Spark
    agg_df = spark.sql(f"""
        SELECT
            AVG(sentiment_score)   AS avg_sentiment,
            COUNT(*)               AS post_count
        FROM social_media
        WHERE LOWER(post_content) LIKE '%' ||
              (SELECT LOWER(company_name) FROM companies WHERE company_id = '{company_id}' LIMIT 1)
              || '%'
    """).toPandas()

    # Recent posts — only 5 rows collected
    posts_df = spark.sql(f"""
        SELECT platform, post_content, sentiment_score, timestamp
        FROM social_media
        WHERE LOWER(post_content) LIKE '%' ||
              (SELECT LOWER(company_name) FROM companies WHERE company_id = '{company_id}' LIMIT 1)
              || '%'
        ORDER BY timestamp DESC
        LIMIT 5
    """).toPandas()

    company_name_df = spark.sql(
        f"SELECT company_name FROM companies WHERE company_id = '{company_id}' LIMIT 1"
    ).toPandas()
    company_name = company_name_df.iloc[0]["company_name"] if not company_name_df.empty else company_id

    if posts_df.empty:
        no_data = html.Div(f"No social media posts found mentioning {company_name}.", className="text-muted")
        return no_data, go.Figure(), html.Div()

    # ── Posts list ──
    posts_section = html.Div([
        html.H6(f"Recent Posts mentioning {company_name} ({int(agg_df.iloc[0]['post_count']):,} total)"),
        html.Ul([
            html.Li([
                html.Strong(f"{r['platform']}  "),
                r["post_content"],
                dbc.Badge(
                    f"Sentiment: {r['sentiment_score']:.2f}",
                    color="success" if r["sentiment_score"] >= 0.5 else "danger",
                    className="ms-2"
                )
            ]) for _, r in posts_df.iterrows()
        ])
    ])

    # ── Gauge ──
    avg = float(agg_df.iloc[0]["avg_sentiment"])
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg,
        number={"valueformat": ".2f"},
        delta={"reference": 0.5, "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0.0, 0.4], "color": "lightcoral"},
                {"range": [0.4, 0.6], "color": "lightyellow"},
                {"range": [0.6, 1.0], "color": "lightgreen"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "value": 0.4}
        },
        title={"text": f"Avg. Social Sentiment<br><sub>{company_name}</sub>"}
    ))

    # ── Fraud keyword flags — counted on Spark, tiny result ──
    fraud_keywords = ["financial fraud", "regulatory breach", "illegitimate transaction",
                      "money laundering", "sanctions"]

    flag_rows = []
    for kw in fraud_keywords:
        count_df = spark.sql(f"""
            SELECT COUNT(*) AS hits
            FROM social_media
            WHERE LOWER(post_content) LIKE '%{kw}%'
              AND LOWER(post_content) LIKE '%' ||
                  (SELECT LOWER(company_name) FROM companies WHERE company_id = '{company_id}' LIMIT 1)
                  || '%'
        """).toPandas()
        hits = int(count_df.iloc[0]["hits"])
        flag_rows.append(
            dbc.ListGroupItem([
                html.Strong(f"{kw.title()}: "),
                dbc.Badge("YES — " + str(hits) + " mention(s)", color="danger")
                if hits > 0 else
                dbc.Badge("No mentions", color="success")
            ])
        )

    flags_section = html.Div([
        html.H6("Compliance & Fraud Keyword Flags", className="mt-3"),
        dbc.ListGroup(flag_rows)
    ])

    return posts_section, gauge_fig, flags_section


# ============================================================
# 7. RUN
# ============================================================
if __name__ == "__main__":
    app.run(debug=False, port=8060)
