from joblib import load
from flask import Flask, request, jsonify, render_template, render_template_string, redirect, url_for
from my_utilities import custom_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import lime
from lime import lime_tabular
matplotlib.use('Agg')

app = Flask(__name__, static_folder='Divers', static_url_path='/Divers')

# Charger le modèle
loaded_data = load('xgb_model.joblib')
model = loaded_data['model']

def save_data(data):
    """Enregistre les données collectées dans un fichier CSV."""

    # Lire les données existantes
    df = pd.read_csv("test_data.csv", index_col=0)

    # Récupère le dernier index et l'incrémente
    last_index = df.index[-1]
    new_index = last_index + 1

    # Ajoutez la nouvelle entrée avec le nouvel index
    df = df.append(pd.Series(data, name=new_index))

    # Enregistrez les données mises à jour
    df.to_csv("test_data.csv")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Récupérer les données du corps de la requête
    data = request.json.get('data')
    
    # Vérifier que les données sont présentes
    if not data:
        return jsonify({"error": "Data not provided"}), 400

    # Transformer les données en DataFrame
    data_df = pd.DataFrame([data])

    # Faire une prédiction
    try:
        output = model.predict(data_df)
        proba = model.predict_proba(data_df)[0][1]
        result = {
            "probabilité": float(proba)
        }
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify(result)

@app.route('/reports')
def show_reports():
    with open("Reports/index.html", 'r') as f:
        content = f.read()
    return render_template_string(content)

@app.route('/report_<report_name>.html')
def show_individual_report(report_name):
    try:
        with open(f"Reports/report_{report_name}.html", 'r', encoding='utf-8') as f:
            content = f.read()
        return render_template_string(content)
    except FileNotFoundError:
        return "Report not found", 404
    
#Créer la route pour le bouton Datadrift
@app.route('/generate_reports', methods=['POST'])
def generate_reports():
    from generate_reports import main as generate_reports_main
    generate_reports_main()
    return redirect(url_for('show_reports'))

#Créer la route pour la collecte de data
@app.route('/collect_data', methods=['POST'])
def collect_and_save_data():

    # Récupération des valeurs
    remboursement_str = request.form.get('REMBOURSEMENT')
    montant_emprunte_str = request.form.get('MONTANT_EMPRUNTE')

    remboursement = float(remboursement_str)
    montant_emprunte = float(montant_emprunte_str)
    
    # Calcul de PAYMENT_RATE
    try:
        payment_rate = remboursement / montant_emprunte
    except ZeroDivisionError:
        return "Erreur : Le montant emprunté ne peut pas être zéro."

    data = {
        "EXT_SOURCE_3": float(request.form.get('EXT_SOURCE_3')),
        "EXT_SOURCE_2": float(request.form.get('EXT_SOURCE_2')),
        "NAME_INCOME_TYPE_Working": int(request.form.get('NAME_INCOME_TYPE_Working')),
        "NAME_EDUCATION_TYPE_Secondary / secondary special": int(request.form.get('NAME_EDUCATION_TYPE_Secondary')),
        "NAME_EDUCATION_TYPE_Higher education": int(request.form.get('NAME_EDUCATION_TYPE_Higher')),
        "OCCUPATION_TYPE_Core staff": int(request.form.get('OCCUPATION_TYPE_Core_staff')),
        "FLAG_DOCUMENT_3": int(request.form.get('FLAG_DOCUMENT_3')),
        "AMT_REQ_CREDIT_BUREAU_HOUR": float(request.form.get('AMT_REQ_CREDIT_BUREAU_HOUR')),
        "CODE_GENDER": int(request.form.get('CODE_GENDER')),
        "PAYMENT_RATE": payment_rate  # Utilisation du taux calculé
    }
    
    # Transformer les données en DataFrame
    data_df = pd.DataFrame([data])

    try:
        proba = model.predict_proba(data_df)[0][1]
        image_path = draw_client(proba)
        lime_image_path = draw_lime(data_df, model)
    except ValueError as e:
        prediction_result = "Erreur lors de la prédiction: {}".format(str(e))

    # Définir la valeur de TARGET en fonction de la probabilité
    data["TARGET"] = 1 if proba >= 0.7 else 0

    # Transformer les données en DataFrame (à nouveau pour inclure la colonne TARGET)
    data_df = pd.DataFrame([data])

    # Enregistre les données
    save_data(data)

    return render_template('home.html', prediction=proba, image_path=image_path)


def draw_client(accuracy_score):
    plt.figure()
    if not np.isscalar(accuracy_score):
        raise ValueError("La valeur fournie à 'draw_gauge' doit être un scalaire.")

    # Normaliser le score entre 0 et 1
    normalized_score = accuracy_score

    # Créer un axe polar
    fig_client, ax_client = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})

    # Dessiner l'arc de la jauge pour la plage de 0 à 70% (en rouge)
    arc_red = np.linspace(0, 0.7 * np.pi, 100)
    r_red = np.full(100, 0.6)
    ax_client.plot(arc_red, r_red, color="black", linewidth=15)

    # Dessiner l'arc de la jauge pour la plage de 70 à 100% (en bleu)
    arc_blue = np.linspace(0.7 * np.pi, np.pi, 100)
    r_blue = np.full(100, 0.6)
    ax_client.plot(arc_blue, r_blue, color="darkorange", linewidth=15)

    # Calculer l'angle pour l'aiguille
    theta = np.pi * (normalized_score)
    needle, = ax_client.plot([theta, theta], [0, 0.6], color='darkorange', linewidth=2)

    # Configurer les angles et étiquettes en pourcentage
    ax_client.set_theta_zero_location("W")
    ax_client.set_theta_direction(-1)
    ax_client.set_xticks(np.radians([0, 45, 90, 126, 135, 180]))
    ax_client.set_xticklabels(["0%", "25%", "50%","70%", "75%", "100%"])
    ax_client.set_yticks([])
    ax_client.set_yticklabels([])
    ax_client.grid(False)
    ax_client.set_ylim([0, 0.6])
    ax_client.set_xlim([0, np.pi])
    ax_client.set_title("EVALUATION DU CLIENT", va="bottom")

     # Ajouter le message "CREDIT APPROUVE" ou "CREDIT REFUSE"
    credit_message = "CREDIT APPROUVE" if accuracy_score >= 0.70 else "CREDIT REFUSE"
    color = "black"
    ax_client.text(0.5, 0.1, credit_message, color=color, ha="center", va="center", transform=ax_client.transAxes, fontsize=20, weight='bold')

    # Ajouter le score en tant que légende
    ax_client.legend([needle], [f'Score = {accuracy_score * 100:.2f}%'])
    
    try:
        image_path = "Divers/jauge_client.png"
        fig_client.savefig(image_path)
        plt.close(fig_client)

    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image : {e}")

    return image_path

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
    

    # Améliorations esthétiques (vous pouvez ajouter/modifier selon vos préférences)
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
    lime_image_path = "Divers/lime_explanation.png"
    fig_lime.tight_layout()
    fig_lime.savefig(lime_image_path)
    plt.close(fig_lime)

    return lime_image_path

if __name__ == "__main__":
    app.run(debug=True)
