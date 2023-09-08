import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, dash_table

df = pd.read_csv("complet_data.csv", index_col="SK_ID_CURR")
df = df.reset_index()

# Initialiser l'application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Layout
app.layout = dbc.Container([
    # Titre
    dbc.Row([
        dbc.Col(html.H1("LE CREDIT QUI DIT OUI", style={'text-align': 'center'}), width={'size': 12, 'offset': 0}, )
    ], className='mb-4'),

    # Sélection des colonnes et sk_id input
    dbc.Row([
                dbc.Col([
            dbc.Input(id='sk_id-input', type='number', value=100002, placeholder='Entrer un sk_id'),
            dbc.Button('Envoi', id='submit-button', color='primary', className='mt-2')
        ], width=1),

        dbc.Col([
            dcc.Dropdown(
                id='column-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                value=df.columns.tolist(),
                multi=True
            )
        ], width=5),

    ], className='mb-4'),

    dbc.Alert(
        "Ce client n'existe pas encore", 
        id="no-id-alert", 
        color="danger", 
        is_open=False,  # Important ! Gardez-le fermé par défaut
        duration=4000,  # L'alerte sera visible pendant 4 secondes
    ),

    dbc.Row([
        dbc.Col(
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_data={
                    'color': 'black',
                    'backgroundColor': 'white'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(220, 220, 220)',
                    }
                ],
                style_header={
                'backgroundColor': '#00CED1',
                'color': 'black',
                'fontWeight': 'bold'}
            ),
            width=6
        )
    ], className='mb-4'),

], fluid=True)

# Callback du tableau
@app.callback(
    Output('table', 'data'),
    Output('table', 'columns'),
    Output('no-id-alert', 'is_open'),
    Input('column-dropdown', 'value'),
    Input('submit-button', 'n_clicks'),
    State('sk_id-input', 'value')
)
def update_table(selected_columns, n_clicks, sk_id=None): 
    if sk_id and sk_id not in df['SK_ID_CURR'].values:
        return dash.no_update, dash.no_update, True

    # Assurez-vous que selected_columns n'est jamais None.
    if not selected_columns:
        selected_columns = df.columns.tolist()

    # Si le sk_id est fourni, filtrez le DataFrame pour obtenir les lignes autour de cette ligne spécifique.
    if sk_id:
        idx = df[df['SK_ID_CURR'] == sk_id].index[0]
        
        # Si le SK_ID est parmi les 9 derniers, alors il est affiché en dernier
        if idx >= len(df) - 9:
            filtered_df = df.iloc[-10:][selected_columns]
        else:
            # Prenez 4 lignes avant l'index, l'index lui-même, puis 5 lignes après
            start = max(0, idx - 4)
            end = start + 10
            filtered_df = df.iloc[start:end][selected_columns]
    else:
        filtered_df = df[selected_columns]

    columns = [{"name": i, "id": i} for i in selected_columns]

    return filtered_df.to_dict('records'), columns, False

if __name__ == '__main__':
    app.run_server(debug=True)
