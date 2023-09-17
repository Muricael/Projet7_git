import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from joblib import load
import requests
import os

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate

import lime
from lime import lime_tabular

from utiles import (calculate_payment_rate, 
                   get_credit_decision, 
                    jauge_pret_plotly,
                   draw_lime)

# Initialiser l'appli
app = Dash(__name__,  suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

df = pd.read_csv("complet_data.csv", index_col="SK_ID_CURR")
df = df.reset_index()

train_df = pd.read_csv('train_data.csv', index_col="SK_ID_CURR")
test_df = pd.read_csv('test_data.csv', index_col="SK_ID_CURR")

test_X =test_df.drop("TARGET", axis = 1)
test_y = test_df['TARGET']

loaded_data = load('xgb_model.joblib')
model = loaded_data['model']


# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("LE CREDIT QUI DIT OUI", style={'text-align': 'center'}), width={'size': 12, 'offset': 0})
    ], className='mb-4'),

    # Sélection des colonnes et sk_id input
    dbc.Row([
        # Colonne de gauche
        dbc.Col([
            # Input SK_ID
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='sk_id-input', type='number', value=100002, placeholder='Entrer un sk_id'),
                    dbc.Button('Envoi', id='submit-button', color='primary', className='mt-2')
                ], width=12)
            ]),
            # Dropdown
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='column-dropdown',
                        options=[{'label': col, 'value': col} for col in df.columns],
                        value=df.columns.tolist(),
                        multi=True,
                        style={"color": "black"}
                    )
                ], width=12)
            ]),
            # Tableau
            dbc.Row([
                dbc.Col(
                    dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_table={'overflowX': 'auto'},
                        style_data={'color': 'black', 'backgroundColor': 'white'},
                        style_data_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(220, 220, 220)'
                        }],
                        style_header={
                            'backgroundColor': '#00CED1',
                            'color': 'black',
                            'fontWeight': 'bold'
                        }
                    ),
                    width=12
                )
            ], className='mb-4')
        ], width=5),

        # Colonne de droite
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Label("Source_3"), width=4),
                dbc.Col(dcc.Input(id='input-source_3', type='number', value=0.0), width=8)
            ], className='mb-2'),
            dbc.Row([
                dbc.Col(html.Label("Source_2"), width=4),
                dbc.Col(dcc.Input(id='input-source_2', type='number', value=0.0), width=8)
            ], className='mb-2'),
            dbc.Row([
                dbc.Col(html.Label("Travail:"), width=4),
                dbc.Col(
                    dcc.RadioItems(
                        id='input-travail',
                        options=[
                            {'label': '  Oui', 'value': True},
                            {'label': '  Non', 'value': False}
                        ],
                        inline=True,
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                    ),
                    width=8
                )
            ], className='mb-2'),
            dbc.Row([
                dbc.Col(html.Label("Lycée"), width=4),
                dbc.Col(
                    dcc.RadioItems(
                        id='input-lycee',
                        options=[
                            {'label': '  Oui', 'value': True},
                            {'label': '  Non', 'value': False}
                        ],
                        inline=True,  
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                    ), 
                    width=8
                )
            ], className='mb-2'),

            dbc.Row([
                dbc.Col(html.Label("Etudes Sup"), width=4),
                dbc.Col(
                    dcc.RadioItems(
                        id='input-etudesup',
                        options=[
                            {'label': '  Oui', 'value': True},
                            {'label': '  Non', 'value': False}
                        ], 
                        inline=True,  
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                    ), 
                    width=8
                )
            ], className='mb-2'),

            dbc.Row([
                dbc.Col(html.Label("Cadre"), width=4),
                dbc.Col(
                    dcc.RadioItems(
                        id='input-cadre',
                        options=[
                            {'label': '  Oui', 'value': True},
                            {'label': '  Non', 'value': False}
                        ], 
                        inline=True,  
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                    ), 
                    width=8
                )
            ], className='mb-2'),

            dbc.Row([
                dbc.Col(html.Label("Indic_doc 3"), width=4),
                dbc.Col(
                    dcc.RadioItems(
                        id='input-indic_doc_3',
                        options=[
                            {'label': '  Oui', 'value': 1},
                            {'label': '  Non', 'value': 0}
                        ],
                        inline=True,  
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                    ), 
                    width=8
                )
            ], className='mb-2'),

            dbc.Row([
                dbc.Col(html.Label("Nb demandes/h"), width=4),
                dbc.Col(dcc.Input(id='input-ndask_per_hour', type='number', value=0), width=8)
                
            ], className='mb-2'),

            dbc.Row([
                dbc.Col(html.Label("Sexe"), width=4),
                dbc.Col(
                    dcc.RadioItems(
                        id='input-sexe',
                        options=[
                            {'label': '  Homme', 'value': 1},
                            {'label': '  Femme', 'value': 0}
                        ],
                        inline=True,  
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                    ), 
                    width=8
                )
            ], className='mb-2'),

            dbc.Row([
                dbc.Col(html.Label("Montant emprunté"), width=4),
                dbc.Col(dcc.Input(id='input-montant_emprunte', type='number', value=0), width=8)
            ], className='mb-2'),
            
            dbc.Row([
                dbc.Col(html.Label("Montant remboursé"), width=4),
                dbc.Col(dcc.Input(id='input-remboursement', type='number', value=0), width=8)
            ], className='mb-2'),

            dbc.Row([
                dbc.Col(dbc.Button('Ajouter un nouveau client', id='new-client-button', color='primary', className='mt-2'), width=8)
            ])

        ], width=4),
        
        dbc.Col([
            dash_table.DataTable(
                id='new-client-table',
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_data={'color': 'black', 'backgroundColor': 'white'},
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(220, 220, 220)'
                }],
                style_header={
                    'backgroundColor': '#00CED1',
                    'color': 'black',
                    'fontWeight': 'bold'
                }
            ),
        ], width=3)

    ], className='mb-4'),

    dbc.Alert(
        "",
        id="credit-decision-alert",
        color="info",
        is_open=False,
        duration=4000
    ),
    

    dbc.Alert(
        "Ce client n'existe pas encore",
        id="no-id-alert",
        color="danger",
        is_open=False,
        duration=4000
    ),


    dbc.Row([
        dbc.Col(html.Img(id='lime_img', src=app.get_asset_url('assets/lime_explanation.png')), width={'size': 5}),
        
        dbc.Col(dcc.Graph(id='my-graph'), width={'size': 3}),
    
        dbc.Col(html.Img(id='shap_img', src=app.get_asset_url('assets/outputshap.png'), style={'width': '80%'}), width={'size': 4}),
    
    ], className='mb-4'),

    dcc.Store(id='prev-button-clicks-store', data=0),
    dcc.Store(id='last-click-store', data=0),



], fluid=True)

@app.callback(
    Output('table', 'data'),
    Output('table', 'columns'),
    Input('column-dropdown', 'value'),
    Input('submit-button', 'n_clicks'),
    State('sk_id-input', 'value')
)
def update_table(selected_columns, n_clicks, sk_id):
    global df

    # Selected_columns n'est jamais None.
    if not selected_columns:
        selected_columns = df.columns.tolist()

    # Si le sk_id est fourni et existe dans le DataFrame
    if sk_id and sk_id in df['SK_ID_CURR'].values:
        row = df[df['SK_ID_CURR'] == sk_id]
        idx = row.index[0]
        start_idx = max(0, idx - 0)
        end_idx = min(len(df) - 1, idx + 6)
        filtered_df = df.iloc[start_idx:end_idx + 1][selected_columns]
        columns = [{"name": i, "id": i} for i in selected_columns]
        return filtered_df.to_dict('records'), columns

    # Si aucune action n'est prise et que la page est simplement chargée.
    filtered_df = df[selected_columns]
    columns = [{"name": i, "id": i} for i in selected_columns]
    return filtered_df.to_dict('records'), columns

@app.callback(
    Output('credit-decision-alert', 'children'),
    Output('credit-decision-alert', 'is_open'),
    Output('new-client-table', 'data'),
    Output('new-client-table', 'columns'),
    Output('last-click-store', 'data'),
    Output('my-graph', 'figure'),
    Output('lime_img', 'src'),
    Output('shap_img', 'src'),
    Input('new-client-button', 'n_clicks'),
    State('input-source_3', 'value'),
    State('input-source_2', 'value'),
    State('input-travail', 'value'),
    State('input-lycee', 'value'),
    State('input-etudesup', 'value'),
    State('input-cadre', 'value'),
    State('input-indic_doc_3', 'value'),
    State('input-ndask_per_hour', 'value'),
    State('input-sexe', 'value'),
    State('input-remboursement', 'value'),
    State('input-montant_emprunte', 'value'),
    State('last-click-store', 'data')
)
def add_new_client(n_clicks, source_3, source_2, travail, lycee, etudesup, 
                   cadre, indic_doc_3, ndask_per_hour, sexe, 
                   remboursement, montant_emprunte, last_click):


    global df
    decision = "Décision non prise"
    new_data = []
    new_columns = []

    if n_clicks is None:
        raise dash.exceptions.PreventUpdate


    # Vérification si le bouton a été cliqué
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    if n_clicks and n_clicks != last_click:
        new_sk_id = max(df['SK_ID_CURR']) + 1 #Crée nouvel index client

        new_row = {
            "SK_ID_CURR": new_sk_id,
            "Source_3": source_3,
            "Source_2": source_2,
            "Travail": bool(travail),
            "Lycée": bool(lycee),
            "Etudes Sup": bool(etudesup),
            "Cadre": bool(cadre),
            "Indic_doc 3": bool(indic_doc_3),
            "Nb demandes/h": int(ndask_per_hour),
            "Sexe": int(sexe),
            "Tx pay." : calculate_payment_rate(remboursement, montant_emprunte)
        }

        # Décision du crédit
        credit_decision, probability = get_credit_decision(model, new_row)
        new_row["TARGET"] = credit_decision
        
        # Chemin lime
        file_path = "assets/lime_explanation.png"

        # Vérifier et suppr. lime_explanation
        if os.path.exists(file_path):
            os.remove(file_path)
            print("Le fichier a été supprimé.")
        else:
            pass

        fig = jauge_pret_plotly(probability)
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df2 = pd.concat([test_df, new_df], ignore_index=True)
        df.to_csv("complet_data.csv", index=False)
        df2.to_csv("test_data.csv", index=False)
        decision = credit_decision
        lime_image = draw_lime(new_df, model)

        
        new_data = [new_row]
        new_columns = [{"name": i, "id": i} for i in new_row.keys()]

        return decision, True, new_data, new_columns, n_clicks, fig, lime_image, app.get_asset_url('outputshap.png')

    return decision, True, new_data, new_columns, n_clicks, dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run(debug=False)
