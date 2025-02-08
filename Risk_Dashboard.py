import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from pyspark.sql import SparkSession
import plotly.express as px
import plotly.graph_objs as go

# Initialize Spark session
spark = SparkSession.builder.appName("RiskDashboard").getOrCreate()

# Load datasets from HDFS using Spark
companies_df = spark.read.csv("hdfs://localhost:9000/user/neeleshkhantwal/Datasets/synthetic_companies_house.csv", header=True, inferSchema=True, sep=",", multiLine=True, escape='"').toPandas()
psc_df = spark.read.csv("hdfs://localhost:9000/user/neeleshkhantwal/Datasets/synthetic_psc_register.csv", header=True, inferSchema=True, sep=",", multiLine=True, escape='"').toPandas()
transactions_df = spark.read.csv("hdfs://localhost:9000/user/neeleshkhantwal/Datasets/synthetic_financial_transactions.csv", header=True, inferSchema=True, sep=",", multiLine=True, escape='"').toPandas()
risk_scores_df = spark.read.csv("hdfs://localhost:9000/user/neeleshkhantwal/Datasets/synthetic_risk_scores.csv", header=True, inferSchema=True, sep=",", multiLine=True, escape='"').toPandas()
social_media_df = spark.read.csv("hdfs://localhost:9000/user/neeleshkhantwal/Datasets/synthetic_social_media_data.csv", header=True, inferSchema=True, sep=",", multiLine=True, escape='"').toPandas()

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Generate the dropdown options again after filtering
company_options = [{'label': name, 'value': company_id} for company_id, name in zip(companies_df['company_id'], companies_df['company_name'])]

# Layout of the app
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Company Structure', children=[
            html.Div([
                dcc.Dropdown(
                    id='company-dropdown',
                    options=company_options,
                    value=company_options[0]['value'],
                    clearable=False,
                ),
                html.Div(id='company-info-content'),
                html.Div(id='psc-info-content')
            ])
        ]),
        dcc.Tab(label='Financial Transactions', children=[
            html.Div(id='financial-transactions-content')
        ]),
        dcc.Tab(label='Risk Dashboard', children=[
            html.Div([
                html.H4("Top Financial Risk Activities"),
                dcc.Graph(id='top-risk-activities-graph'),
                html.Div([
                    html.Div([
                        html.H4("Social Media Sentiment Analysis"),
                        html.Div(id='recent-social-media-posts-content'),
                    ], style={'width': '55%', 'display': 'inline-block'}),
                    html.Div([
                        dcc.Graph(id='social-media-sentiment-meter'),
                    ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ])
            ])
        ])
    ])
])

# Callbacks to update the content based on the selected company
@app.callback(
    Output('company-info-content', 'children'),
    Input('company-dropdown', 'value')
)
def update_company_info(company_id):
    # Filter the company data for the selected company
    company_info = companies_df[companies_df['company_id'] == company_id].iloc[0]
    
    # Create a div with company information
    company_info_div = html.Div([
        html.H4(f"Company Information for {company_info['company_name']}"),
        html.P(f"Registration Number: {company_info['registration_number']}"),
        html.P(f"Incorporation Date: {company_info['incorporation_date']}"),
        html.P(f"Registered Address: {company_info['registered_address']}"),
        html.P(f"Nature of Business: {company_info['nature_of_business']}"),
        html.P(f"Status: {company_info['status']}")
    ])
    
    return company_info_div

@app.callback(
    Output('psc-info-content', 'children'),
    Input('company-dropdown', 'value')
)
def update_psc_info(company_id):
    # Filter the PSC data for the selected company
    psc_info = psc_df[psc_df['company_id'] == company_id]
    
    if not psc_info.empty:
        # Create a table with PSC information
        table = dbc.Table.from_dataframe(psc_info[['psc_id', 'psc_name', 'date_of_birth', 'nationality', 'psc_address', 'control_type']], striped=True, bordered=True, hover=True)
    else:
        table = html.Div("No PSC data available for this company.")
    
    return html.Div([
        html.H4(f"Persons with Significant Control (PSC) for {companies_df[companies_df['company_id'] == company_id]['company_name'].values[0]}"),
        table
    ])

@app.callback(
    Output('financial-transactions-content', 'children'),
    Input('company-dropdown', 'value')
)
def update_financial_transactions(company_id):
    # Filter transactions data for the selected company
    filtered_transactions = transactions_df[transactions_df['company_id'] == company_id]
    
    # Generate the transactions table
    return dbc.Table.from_dataframe(filtered_transactions, striped=True, bordered=True, hover=True)

@app.callback(
    Output('top-risk-activities-graph', 'figure'),
    Input('company-dropdown', 'value')
)
def update_top_risk_activities(company_id):
    # Filter risk scores data for the selected company
    filtered_risks = risk_scores_df[risk_scores_df['company_id'] == company_id].sort_values(by='transaction_date', ascending=True)
    
    if not filtered_risks.empty:
        # First, extract the company name separately
        company_name = companies_df[companies_df['company_id'] == company_id]["company_name"].values[0]
        # Then use the extracted company name in your f-string
        fig = px.line(filtered_risks, x='transaction_date', y='risk_score', title=f'Risk Scores Over Time for {company_name}')
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=filtered_risks['transaction_date'].min(), x1=filtered_risks['transaction_date'].max(),
            y0=0.7, y1=0.7,
            line=dict(color="Red", width=2, dash="dash"),
        )
    else:
        fig = go.Figure()

    return fig

# Ensure all company names are in lowercase for easier matching
companies_df['company_name'] = companies_df['company_name'].str.lower()

# Define sentiment keywords and their corresponding scores
sentiment_keywords = {
    'terrible': 0.0,
    'bad': 0.2,
    'not good': 0.3,
    'okay': 0.5,
    'good': 0.6,
    'great': 0.7,
    'amazing': 0.8,
    'excellent': 0.9,
    'phenomenal': 1.0
}

# Function to find the company name in the post content and return the company_id
def match_company_id(post_content):
    if isinstance(post_content, str):  # Ensure post_content is a string
        post_content = post_content.lower()
        for _, row in companies_df.iterrows():
            if row['company_name'] in post_content:
                return row['company_id']
    return None

# Function to assign sentiment score based on keywords in the post content
def assign_sentiment_score(post_content):
    if isinstance(post_content, str):
        post_content = post_content.lower()
        for keyword, score in sentiment_keywords.items():
            if keyword in post_content:
                return score
    return 0.5  # Default to neutral if no keywords are found

# Apply the function to match and assign the company_id
social_media_df['company_id'] = social_media_df['post_content'].apply(match_company_id)

# Apply the function to assign sentiment score
social_media_df['social_media_sentiment'] = social_media_df['post_content'].apply(assign_sentiment_score)

# Save the updated social media data back to HDFS
updated_social_media_df = spark.createDataFrame(social_media_df)
updated_social_media_df.write.csv(
    "hdfs://localhost:9000/data/social_media/updated_social_media_data.csv",
    header=True,
    mode='overwrite'
)

@app.callback(
    [Output('recent-social-media-posts-content', 'children'),
     Output('social-media-sentiment-meter', 'figure')],
    Input('company-dropdown', 'value')
)
def update_social_media_posts_and_sentiment(company_id):
    # Normalize the company_id
    company_id = company_id.strip().lower()

    # Get company name
    company_row = companies_df[companies_df['company_id'] == company_id]
    if company_row.empty:
        return html.Div("Company not found."), go.Figure()

    company_name = company_row['company_name'].values[0]
    
    # Filter social media data for the selected company
    filtered_social_media = social_media_df[social_media_df['company_id'] == company_id]
    
    if filtered_social_media.empty:
        return html.Div(f"No social media data available for {company_name}."), go.Figure()

    # Display the most recent posts (limit to 5 for brevity) with sentiment scores
    recent_posts = filtered_social_media.sort_values(by='timestamp', ascending=False).head(5)
    post_content = html.Div([
        html.H4(f"Recent Social Media Posts for {company_name}"),
        html.Ul([html.Li(f"{row['platform']} - {row['post_content']} (Sentiment Score: {row['social_media_sentiment']:.2f})")
                 for _, row in recent_posts.iterrows()])
    ])

    # Calculate the average sentiment score
    avg_sentiment = filtered_social_media['social_media_sentiment'].mean()

    # Create the sentiment meter
    sentiment_meter = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_sentiment,
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "green" if avg_sentiment > 0.5 else "red"},
            'steps': [
                {'range': [0, 0.5], 'color': "lightcoral"},
                {'range': [0.5, 1], 'color': "lightgreen"}
            ],
        },
        title={'text': f"Social Media Sentiment for {company_name}"}
    ))

    return post_content, sentiment_meter

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, port=8060)
