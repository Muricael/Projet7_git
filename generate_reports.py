import pandas as pd
import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnSummaryMetric, ColumnQuantileMetric, ColumnDriftMetric

def clean_column_name(column_name):
    return column_name.replace("/", "_").replace(" ", "_").replace("\\", "_")

def main():
    # Load the dataframes
    train_df = pd.read_html('train_data.html')[0]
    test_df = pd.read_html('test_data.html')[0]

    # Drop the first column (assuming it's an unwanted index column)
    train_df = train_df.drop(train_df.columns[0], axis=1)
    test_df = test_df.drop(test_df.columns[0], axis=1)

    # Create a column mapping
    column_mapping = ColumnMapping(
        target='TARGET', 
        numerical_features=list(train_df.columns)  # Ensure all columns are numerical here.
    )

    columns_to_analyze = [col for col in train_df.columns if col != 'SK_ID_CURR']
    reports = {}

    for column in columns_to_analyze:
        report = Report(metrics=[
            ColumnSummaryMetric(column_name=column),
            ColumnQuantileMetric(column_name=column, quantile=0.25),
            ColumnDriftMetric(column_name=column),
        ])
        
        report.run(reference_data=train_df, current_data=test_df)
        reports[column] = report

    for column, report in reports.items():
        print(f"Report for column: {column}")
        cleaned_column_name = clean_column_name(column)
        report_path = f"report_{cleaned_column_name}.html"
        report.save_html(report_path)
        print(f"Report saved as {report_path}")
        print("-" * 50)

if __name__ == "__main__":
    main()



#espérons que ça remarche après relancement