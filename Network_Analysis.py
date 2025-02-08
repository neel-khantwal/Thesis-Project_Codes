import pandas as pd
import dash
from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objs as go

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load and combine the datasets
def load_data():
    try:
        # Load data files from the provided paths
        nodes_address_df = pd.read_csv('/Users/neeleshkhantwal/Documents/Thesis/Data/Panama/panama_papers.nodes.address.csv', low_memory=False)
        nodes_entity_df = pd.read_csv('/Users/neeleshkhantwal/Documents/Thesis/Data/Panama/panama_papers.nodes.entity.csv', low_memory=False)
        nodes_intermediary_df = pd.read_csv('/Users/neeleshkhantwal/Documents/Thesis/Data/Panama/panama_papers.nodes.intermediary.csv', low_memory=False)
        nodes_officer_df = pd.read_csv('/Users/neeleshkhantwal/Documents/Thesis/Data/Panama/panama_papers.nodes.officer.csv', low_memory=False)
        nodes_other_df = pd.read_csv('/Users/neeleshkhantwal/Documents/Thesis/Data/Panama/panama_papers.nodes.other.csv', low_memory=False)
        edges_df_full = pd.read_csv('/Users/neeleshkhantwal/Documents/Thesis/Data/Panama/panama_papers.edges.csv', low_memory=False)
        
        # Combine all node data into one DataFrame
        nodes_combined_df = pd.concat([nodes_address_df, nodes_entity_df, nodes_intermediary_df, nodes_officer_df, nodes_other_df])
        
        # Replace empty strings or NaN values with "N/A"
        nodes_combined_df.fillna("N/A", inplace=True)
        nodes_combined_df.replace("", "N/A", inplace=True)
        
        return nodes_combined_df, edges_df_full
    except Exception as e:
        return f"An error occurred while loading data: {str(e)}", None

# Load data once globally to avoid reloading on each callback
nodes_data, edges_data = load_data()

# Get the list of unique entities for the dropdown menu
entities = nodes_data["n.name"].unique()

# Define the layout of the Dash app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Network Analysis Dashboard", className="display-4 text-center text-primary mt-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='entity-dropdown',
                options=[{'label': entity, 'value': entity} for entity in entities],
                placeholder="Select an Entity",
                className="mb-4"
            ),
            width=6,
            className="offset-md-3"
        )
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='entity-details'), width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(html.Div(id='connections-section'), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='network-graph', style={'height': '600px'}), width=12)
    ])
], fluid=True)

# Callback to update entity details when an entity is selected
@app.callback(
    Output('entity-details', 'children'),
    Input('entity-dropdown', 'value')
)
def update_entity_details(selected_entity):
    if selected_entity is None:
        return html.Div("Select an entity to see the details.", className="text-center text-muted")

    # Filter the data for the selected entity
    entity_data = nodes_data[nodes_data['n.name'] == selected_entity]

    # Display relevant information about the entity
    details = dbc.Card([
        dbc.CardHeader(html.H4(f"Details for {selected_entity}", className="card-title")),
        dbc.CardBody([
            html.P(f"Address: {entity_data['n.address'].values[0]}", className="card-text"),
            html.P(f"Jurisdiction Description: {entity_data['n.jurisdiction_description'].values[0]}", className="card-text"),
            html.P(f"Company Type: {entity_data['n.company_type'].values[0]}", className="card-text"),
            html.P(f"Status: {entity_data['n.status'].values[0]}", className="card-text"),
            html.P(f"Other Information: {entity_data['n.note'].values[0]}", className="card-text")
        ])
    ], className="shadow-sm mb-4")

    return details

# Callback to update the connections section when an entity is selected
@app.callback(
    Output('connections-section', 'children'),
    Input('entity-dropdown', 'value')
)
def update_connections(selected_entity):
    if selected_entity is None:
        return html.Div()

    # Get the node_id for the selected entity
    entity_node_id = nodes_data[nodes_data['n.name'] == selected_entity]['n.node_id'].values[0]

    # Find connections in the edges data where the selected entity is either node_1 or node_2
    connections = edges_data[(edges_data['node_1'] == entity_node_id) | (edges_data['node_2'] == entity_node_id)]

    if connections.empty:
        return html.Div("No connections found for this entity.", className="text-center text-muted")

    # Prepare the data for the table
    connection_rows = []
    for _, row in connections.iterrows():
        connected_node_id = row['node_1'] if row['node_2'] == entity_node_id else row['node_2']
        connected_entity_data = nodes_data[nodes_data['n.node_id'] == connected_node_id]
        
        connection_rows.append({
            "Connected Entity": connected_entity_data['n.name'].values[0],
            "Address": connected_entity_data['n.address'].values[0],
            "Jurisdiction": connected_entity_data['n.jurisdiction_description'].values[0],
            "Company Type": connected_entity_data['n.company_type'].values[0],
            "Status": connected_entity_data['n.status'].values[0],
            "Additional Info": connected_entity_data['n.note'].values[0]
        })

    # Remove duplicate entries
    connection_df = pd.DataFrame(connection_rows).drop_duplicates()

    # Create a table to display the connections
    connection_table = dash_table.DataTable(
        columns=[
            {"name": "Connected Entity", "id": "Connected Entity"},
            {"name": "Address", "id": "Address"},
            {"name": "Jurisdiction", "id": "Jurisdiction"},
            {"name": "Company Type", "id": "Company Type"},
            {"name": "Status", "id": "Status"},
            {"name": "Additional Info", "id": "Additional Info"}
        ],
        data=connection_df.to_dict('records'),
        style_table={'overflowX': 'auto', 'boxShadow': '0px 0px 5px 0px rgba(0,0,0,0.2)'},
        style_cell={'textAlign': 'left', 'padding': '10px', 'whiteSpace': 'normal', 'height': 'auto'},
        style_header={
            'backgroundColor': '#007BFF',
            'fontWeight': 'bold',
            'color': 'white',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f9f9f9'
            },
            {
                'if': {'column_id': 'Status', 'filter_query': '{Status} = "Defaulted"'},
                'backgroundColor': '#FFCCCB',
                'color': 'black'
            }
        ],
    )

    return dbc.Card([
        dbc.CardHeader(html.H4("Connections", className="card-title")),
        dbc.CardBody(connection_table)
    ], className="shadow-sm")

# Callback to create the network graph with enhanced visualization and additional information
@app.callback(
    Output('network-graph', 'figure'),
    Input('entity-dropdown', 'value')
)
def update_network_graph(selected_entity):
    if selected_entity is None:
        return {}

    # Create a network graph
    G = nx.Graph()

    # Add nodes and edges to the graph based on connections
    entity_node_id = nodes_data[nodes_data['n.name'] == selected_entity]['n.node_id'].values[0]
    connections = edges_data[(edges_data['node_1'] == entity_node_id) | (edges_data['node_2'] == entity_node_id)]
    
    for _, row in connections.iterrows():
        node1_id = row['node_1']
        node2_id = row['node_2']
        node1_name = nodes_data[nodes_data['n.node_id'] == node1_id]['n.name'].values[0]
        node2_name = nodes_data[nodes_data['n.node_id'] == node2_id]['n.name'].values[0]
        
        G.add_edge(node1_name, node2_name, weight=row.get('transaction_amount', 1))  # Use transaction amount if available

    # Calculate centrality measures
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)

    # Adjust spring layout to bring nodes closer
    pos = nx.spring_layout(G, k=0.3, iterations=50)  # Adjust k value to control spacing

    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=edge[2]['weight'] * 2, color='#888'),  # Adjust width by weight
            hoverinfo='none',
            mode='lines'))

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[20 * betweenness[node] for node in G.nodes()],  # Scale by betweenness centrality
            color=[eigenvector[node] for node in G.nodes()],  # Color by eigenvector centrality
            colorbar=dict(
                thickness=15,
                title='Eigenvector Centrality',
                xanchor='left',
                titleside='right'
            ),
        ),
        textfont=dict(size=14)
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        # Add centrality measures and risk levels to the hover text
        hover_text = (
            f"{node}<br>Betweenness: {betweenness[node]:.2f}<br>Closeness: {closeness[node]:.2f}<br>"
            f"Eigenvector: {eigenvector[node]:.2f}<br>"
            f"Jurisdiction Risk Level: {nodes_data[nodes_data['n.name'] == node]['n.jurisdiction_description'].values[0]}"
        )
        node_trace['text'] += tuple([hover_text])

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title=f"Network Graph for {selected_entity}",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper") ],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        height=600
                    ))
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, port=8070)
