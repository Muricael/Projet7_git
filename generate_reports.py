import pandas as pd
import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import ColumnSummaryMetric, ColumnQuantileMetric, ColumnDriftMetric
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

def clean_column_name(column_name):
    # Remplacer les caractères non autorisés par '_'
    return column_name.replace("/", "_").replace(" ", "_").replace("\\", "_")

# Chargement des données depuis les fichiers HTML
train_df = pd.read_html('train_data.html')[0]
test_df = pd.read_html('test_data.html')[0]

# Suppression de la première colonne (index ajouté lors de la sauvegarde en HTML)
train_df = train_df.drop(train_df.columns[0], axis=1)
test_df = test_df.drop(test_df.columns[0], axis=1)

# Configuration des colonnes
column_mapping = ColumnMapping(
    target='TARGET', 
    numerical_features=list(train_df.columns)  # assurant que toutes vos colonnes sont numériques ici
)

# List of columns to analyze (en supposant que 'SK_ID_CURR' est une colonne que vous ne voulez pas analyser)
columns_to_analyze = [col for col in train_df.columns if col != 'SK_ID_CURR']

# Dictionary to store reports for each column
reports = {}

for column in columns_to_analyze:
    report = Report(metrics=[
        ColumnSummaryMetric(column_name=column),
        ColumnQuantileMetric(column_name=column, quantile=0.25),
        ColumnDriftMetric(column_name=column),
    ])
    
    report.run(reference_data=train_df, current_data=test_df)
    reports[column] = report

# Affichage et sauvegarde des rapports
for column, report in reports.items():
    print(f"Report for column: {column}")
    cleaned_column_name = clean_column_name(column)
    report_path = f"report_{cleaned_column_name}.html"
    report.save_html(report_path)
    print(f"Report saved as {report_path}")
    print("-"*50)