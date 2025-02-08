import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from tensorflow.keras import layers, models

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 3
data_normal = np.random.normal(loc=50, scale=5, size=(n_samples, n_features))
data_anomalous = np.random.normal(loc=70, scale=5, size=(50, n_features))
data = np.vstack([data_normal, data_anomalous])

df = pd.DataFrame(data, columns=['Transaction Amount', 'Account Balance', 'Transaction Frequency'])
df = df.sample(frac=1).reset_index(drop=True)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['IsoForest_Anomaly'] = iso_forest.fit_predict(df)
df['IsoForest_Anomaly'] = df['IsoForest_Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Autoencoder
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Transaction Amount', 'Account Balance', 'Transaction Frequency']])
input_dim = df_scaled.shape[1]
encoding_dim = 2
input_layer = layers.Input(shape=(input_dim,))
encoder = layers.Dense(encoding_dim, activation="relu")(input_layer)
decoder = layers.Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = models.Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(df_scaled, df_scaled, epochs=50, batch_size=16, shuffle=True, validation_split=0.2, verbose=0)
reconstructions = autoencoder.predict(df_scaled)
reconstruction_error = np.mean(np.abs(reconstructions - df_scaled), axis=1)
threshold = np.percentile(reconstruction_error, 95)
df['Autoencoder_Anomaly'] = (reconstruction_error > threshold).astype(int)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan.fit(df_scaled)
df['DBSCAN_Anomaly'] = (dbscan.labels_ == -1).astype(int)

# Create Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Anomaly Detection Dashboard"),

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Suspicious Transaction Isolation', value='tab-1'),
        dcc.Tab(label='Anomaly Detection via Transaction Patterns', value='tab-2'),
        dcc.Tab(label='Density-Based Anomaly Detection', value='tab-3'),
    ]),

    html.Div(id='tabs-content')
])

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    method_titles = {
        'tab-1': 'Suspicious Transaction Isolation',
        'tab-2': 'Anomaly Detection via Transaction Patterns',
        'tab-3': 'Density-Based Anomaly Detection'
    }

    if tab == 'tab-1':
        selected_method = 'IsoForest_Anomaly'
    elif tab == 'tab-2':
        selected_method = 'Autoencoder_Anomaly'
    else:
        selected_method = 'DBSCAN_Anomaly'

    # Create the main anomaly detection graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Transaction Amount'], mode='markers',
                             marker=dict(color=df[selected_method], colorscale='Bluered', size=8)))
    fig.update_layout(title=method_titles[tab],
                      xaxis_title="Transaction Index",
                      yaxis_title="Transaction Amount",
                      yaxis_range=[40, 80])

    # Create a table of anomalous transactions
    anomalous_df = df[df[selected_method] == 1]
    anomalous_df['Transaction Amount'] = anomalous_df['Transaction Amount'].apply(lambda x: f"{x:.2f} €")
    anomalous_df['Account Balance'] = anomalous_df['Account Balance'].apply(lambda x: f"{x:.2f} €")
    anomalous_df['Transaction Frequency'] = anomalous_df['Transaction Frequency'].round().astype(int)
    anomalous_table = go.Figure(data=[go.Table(
        header=dict(values=['Index', 'Transaction Amount', 'Account Balance', 'Transaction Frequency'],
                    fill_color='paleturquoise',
                    align='center',
                    font=dict(size=20)),
        cells=dict(values=[anomalous_df.index, anomalous_df['Transaction Amount'], anomalous_df['Account Balance'], anomalous_df['Transaction Frequency']],
                   fill_color='lavender',
                   align='center',
                   height=30,  # Adjust the row height
                   font=dict(size=18))
    )])
    anomalous_table.update_layout(title="Anomalous Transactions Detected", height=1200)  # Adjust the table height for better visualization

    # Create the synthetic dataset overview scatter matrix
    overview_fig = px.scatter_matrix(df, dimensions=['Transaction Amount', 'Account Balance', 'Transaction Frequency'],
                                     color=selected_method, title="Overview of Synthetic Financial Dataset",
                                     color_continuous_scale=px.colors.sequential.Bluered)
    overview_fig.update_traces(marker=dict(size=5))
    overview_fig.update_layout(height=800, width=1200, margin=dict(l=20, r=20, t=50, b=50))

    return html.Div([
        dcc.Graph(figure=fig),
        html.H2("List of Anomalous Transactions"),
        dcc.Graph(figure=anomalous_table),
        html.H2("Synthetic Dataset Overview"),
        dcc.Graph(figure=overview_fig)
    ])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=False, port=8090)
