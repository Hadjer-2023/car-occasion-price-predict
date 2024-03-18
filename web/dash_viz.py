
import os

import dash
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State

import numpy as np 
import pandas as pd
import plotly.express as pxt
import plotly.graph_objs as go

basedir = os.path.abspath(os.path.dirname(__file__))
data = os.path.join(basedir, "../data/train.csv")

app = dash.Dash(__name__)

df = pd.read_csv(data)
df['Name'] = df['Name'].str.split().str[0]

# brands = np.sort(df['Name'].unique(), kind='mergesort')[::-1]
brands = np.sort(df['Name'].unique())
years = np.sort(df['Year'].unique(), kind='mergesort')[::-1]
fuels = np.sort(df['Fuel_Type'].unique(), kind='mergesort')[::-1]

selected_brand = 'Maruti'
selected_year = 2018
filtered_df = df.query("Name == @selected_brand and Year == @selected_year")

trace = go.Scatter(
    x=filtered_df['Kilometers_Driven'],
    y=filtered_df['Price'],
    mode='markers'
)

data = [trace]

layout = go.Layout(
    title='Évolution du prix en fonction du kilométrage pour {} {}'.format(selected_brand, selected_year),
    xaxis_title='Kilomètres parcourus',
    yaxis_title='Prix (en euros)'
)

fig = go.Figure(data=data, layout=layout)

app.layout = html.Div([
    html.H2('Évolution du prix en fonction de la marque et de l\'année'),
    html.Div([
        html.Label('Marque : '),
        dcc.Dropdown(
            id='brand-dropdown',
            options=[{'label': brand, 'value': brand} for brand in brands],
            value=selected_brand
        )
    ]),
    html.Div([
        html.Label('Année : '),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': year, 'value': year} for year in years],
            value=selected_year
        )
    ]),
    dcc.Graph(id='price-chart', figure=fig)
])

@app.callback(
    dash.dependencies.Output('price-chart', 'figure'),
    [dash.dependencies.Input('brand-dropdown', 'value'),
     dash.dependencies.Input('year-dropdown', 'value')])
def update_chart(selected_brand, selected_year):
    filtered_df = df.query("Name == @selected_brand and Year == @selected_year")

    trace = go.Scatter(
        x=filtered_df['Kilometers_Driven'],
        y=filtered_df['Price'],
        mode='markers'
    )

    data = [trace]

    layout = go.Layout(
        title='Évolution du prix en fonction du kilométrage pour {} {}'.format(selected_brand, selected_year),
        xaxis_title='Kilomètres parcourus',
        yaxis_title='Prix (en euros)'
    )

    return go.Figure(data=data, layout=layout)

if __name__ == '__main__':
    app.run_server(debug=False)
