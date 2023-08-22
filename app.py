from joblib import load
from flask import Flask, request, jsonify, render_template, render_template_string, redirect, url_for
from my_utilities import custom_loss
import numpy as np
import pandas as pd

app = Flask(__name__)

# Charger le modèle
loaded_data = load('xgb_model.joblib')
model = loaded_data['model']

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
    
    print("Valeur de 2 REMBOURSEMENT:", remboursement_str)
    print("Valeur de 2 MONTANT_EMPRUNTE:", montant_emprunte_str)
    
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

    # Faire une prédiction
    try:
        proba = model.predict_proba(data_df)[0][1]
        prediction_result = "Probabilité : {:.2f}%".format(proba*100)
    except ValueError as e:
        prediction_result = "Erreur lors de la prédiction: {}".format(str(e))
    
    return render_template('home.html', prediction=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
