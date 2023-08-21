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
def home():
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
            "prediction": int(output[0]),
            "probability_class_1": float(proba)
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
    
#Créer la route pour le bouton 
@app.route('/generate_reports', methods=['POST'])
def generate_reports():
    from generate_reports import main as generate_reports_main
    generate_reports_main()
    return redirect(url_for('show_reports'))


if __name__ == "__main__":
    app.run(debug=True)
