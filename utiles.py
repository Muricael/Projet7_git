import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import lime
from lime import lime_tabular
import plotly.graph_objects as go
import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnSummaryMetric, ColumnQuantileMetric, ColumnDriftMetric
import os

def calculate_payment_rate(remboursement, montant_emprunte):
    try:
        payment_rate = remboursement / montant_emprunte
    except ZeroDivisionError:
        if remboursement != 0:
            return "Erreur : Le montant emprunté ne peut pas être zéro."
        else:
            payment_rate = 0  # Remboursement et montant emprunté sont tous les deux 0
    return payment_rate

def get_credit_decision(model, new_data):
    new_df = pd.DataFrame([new_data])
    new_df = new_df.drop(columns=['SK_ID_CURR'])
    probability = model.predict_proba(new_df)[0][1]
    decision = "Approuvé!" if probability > 0.7 else "Refusé!"

    return decision, probability


def jauge_pret_plotly(predic):
    width_of_exterior = 20  # Épaisseur des contours rouges et bleus

    # Contour rouge (0-70%)
    theta_red = [i for i in range(0, 126)]
    r_red = [0.61 for _ in theta_red]
    trace_red = go.Scatterpolar(
        r=r_red,
        theta=theta_red,
        mode='lines',
        line=dict(color="red", width=width_of_exterior)
    )

    # Contour bleu (70-100%)
    theta_blue = [i for i in range(126, 180)]
    r_blue = [0.61 for _ in theta_blue]
    trace_blue = go.Scatterpolar(
        r=r_blue,
        theta=theta_blue,
        mode='lines',
        line=dict(color="darkblue", width=width_of_exterior)
    )

    theta_needle = 180 * predic
    trace_needle = go.Scatterpolar(
        r=[0, 0.59],
        theta=[theta_needle, theta_needle],
        mode="lines",
        line=dict(color="darkred", width=2)
    )

    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                showline=False,
                showticklabels=False,
                ticks='',
                range=[0, 0.7]
            ),
            angularaxis=dict(
                tickvals=[0, 45, 90, 135, 180],
                ticktext=["0%", "25%", "50%", "75%", "100%"],
                direction="clockwise",
                rotation=180,
                showline=True,
                showgrid=True,
                tickcolor="white",  # Coloration en blanc
                tickfont=dict(color="white")  # Coloration en blanc
            ),
            sector=[0, 180]
        ),
        paper_bgcolor="rgba(255,255,255,0)",  
        showlegend=False,
        title=dict(text="EVALUATION DU CLIENT", font=dict(color="white")),  # Coloration en blanc
        annotations=[dict(
            text=f'Probabilité: {predic*100:.2f}%',
            font=dict(color="white"),  # Coloration en blanc
            xref='paper',
            yref='paper',
            x=0.5,
            y=1.15,
            showarrow=False
        )]
    )

    fig = go.Figure(data=[trace_red, trace_blue, trace_needle], layout=layout)
    return fig

def draw_lime(data_df, model):
    print("Fonction draw_lime appelée")
    tables = pd.read_csv('train_data.csv', index_col="SK_ID_CURR")
    X_train = tables.drop('TARGET', axis=1)
    y_train = tables['TARGET']

    explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values,
                                                feature_names=X_train.columns.tolist(),
                                                class_names=['0', '1'],
                                                verbose=True, 
                                                mode='classification')
    sample_data = X_train.iloc[0].values
    try:
        exp = explainer.explain_instance(sample_data, model.predict_proba, num_features=10)
        print("LIME explanation generated successfully.")
    except Exception as e:
        print(f"Error during LIME explanation: {e}")

    # Utilisation de la méthode as_pyplot_figure pour générer le graphique
    fig_lime = exp.as_pyplot_figure()

    # Ajuster la taille de la figure
    fig_lime.set_size_inches(10, 4)

    # Récupération des dimensions actuelles et ajustement de la taille
    current_width, current_height = fig_lime.get_size_inches()
    fig_lime.set_size_inches(current_width * 0.8, current_height * 0.8)
    

    # Améliorations esthétiques
    ax = fig_lime.gca()
    ax.set_facecolor('lightgray')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_title("Détail du prêt")

        # Changer les couleurs des barres
    bars = ax.patches
    for bar in bars:
        if bar.get_facecolor()[0] == 1.0:  # Supposant que le rouge est représenté par (1, 0, 0, 1)
            bar.set_facecolor('black')
        else:
            bar.set_facecolor('orange')

    # Sauvegarder le graphique dans le dossier "Divers"
    lime_image_path = "assets/lime_explanation.png"
    try:
        fig_lime.tight_layout()
        fig_lime.savefig(lime_image_path)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image : {e}")
    plt.close(fig_lime)

    return lime_image_path