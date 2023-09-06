import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from dash import Dash, html, Input, Output, callback, ctx, dash_table

df = pd.read_csv("complet_data.csv", index_col = "SK_ID_CURR")

app = Dash(__name__)

app.layout = dash_table.DataTable(df.to_dict('records'))

if __name__ == '__main__':
    app.run_server(debug=True)