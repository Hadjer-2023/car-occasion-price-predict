
import os

import dash
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import numpy as np 
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

basedir = os.path.abspath(os.path.dirname(__file__))
data = os.path.join(basedir, "../data/train.csv")

df = pd.read_csv(data)
df["Name"] = df["Name"].str.split(" ").str[0]

# convert price from lakhs to indian roupies
conversion_rate = 87
lakh_conversion = 100000
df["Price_Rupie"] = df["Price"] * lakh_conversion

# convert price in euros
df["Price_EUR"] = df["Price_Rupie"] / conversion_rate
df["Price_EUR"] = df["Price_EUR"].round(2)

app = dash.Dash(__name__)

dropdown_style = {
    'width': '350px',
    'margin-top': '4px',
    'margin-bottom': '20px',
    'font-size': '16px'
}

label_style = {
    'font-size': '18px',    
    'font-weight': 'bold',       
    'color': 'brown',          
    'margin-top': '20px'
}

app.layout = html.Div([

        html.Div([
        html.Div([
            html.Div([
                html.Label("Types de carburant :", style=label_style),
                dcc.Dropdown(
                    id="fuel_2",
                    options=[{"label": x, "value": x} for x in df["Fuel_Type"].unique()],
                    placeholder="Sélection",
                    value="Diesel",
                    style=dropdown_style
                )
            ]),
            html.Div([
                html.Label("Année de départ :", style=label_style),
                dcc.Dropdown(
                    id="start_year_2",
                    options=[{"label": x, "value": x} for x in sorted(df["Year"].unique(), reverse=True)],
                    value=df["Year"].min(),
                    style=dropdown_style
                )
            ]),
            html.Div([
                html.Label("Année de fin :", style=label_style),
                dcc.Dropdown(
                    id="end_year_2",
                    options=[{"label": x, "value": x} for x in sorted(df["Year"].unique(), reverse=True)],
                    value=df["Year"].max(),
                    style=dropdown_style
                )
            ]),
            html.Div([
                html.Label("Marque :", style=label_style),
                dcc.Dropdown(
                    id="name_2",
                    options=[{"label": x, "value": x} for x in df["Name"].unique()],
                    value="All",
                    style=dropdown_style
                )
            ])
        ], style={
            'width': '25%',
            'display': 'flex',
            'flex-direction': 'column',
            'align-items': 'center',
            'justify-content': "center"
        }),
        html.Div([
            dcc.Graph(
                id="graph_2",
                  style={
                    'width': '90%',
                    'display': 'flex',
                    'flex-direction': 'row',
                    'justify-content': 'center'
                }
            )
        ], style={
            'width': '75%',
            'display': 'flex',
            'flex-direction': 'column',
            'align-items': 'center',
            'margin-top': '80px'
        })
    ], style={
        'width': '100%',
        'display': 'flex',
        'flex-direction': 'row',
        'justify-content': 'center'
    }),

    html.Div([
        html.Div([
            html.Div([
                html.Label("Types de carburant :", style=label_style),
                dcc.Dropdown(
                    id="fuel_1",
                    options=[{"label": x, "value": x} for x in df["Fuel_Type"].unique()],
                    placeholder="Sélection",
                    value="Diesel",
                    style=dropdown_style
                )
            ]),
            html.Div([
                html.Label("Année de départ :", style=label_style),
                dcc.Dropdown(
                    id="start_year_1",
                    options=[{"label": x, "value": x} for x in sorted(df["Year"].unique(), reverse=True)],
                    value=df["Year"].max(),
                    style=dropdown_style
                )
            ]),
            html.Div([
                html.Label("Année de fin :", style=label_style),
                dcc.Dropdown(
                    id="end_year_1",
                    options=[{"label": x, "value": x} for x in sorted(df["Year"].unique(), reverse=True)],
                    value=df["Year"].max(),
                    style=dropdown_style
                )
            ]),
            # html.Div([
            #     html.Label("Marque :"),
            #     dcc.Dropdown(
            #         id="name_1",
            #         options=[{"label": x, "value": x} for x in df["Name"].unique()],
            #         value="All",
            #         style=dropdown_style
            #     )
            # ])
        ], style={
            'width': '25%',
            'display': 'flex',
            'flex-direction': 'column',
            'align-items': 'center',
            'justify-content': "center"
        }),
        html.Div([
            dcc.Graph(
                id="graph_1",
                  style={
                    'width': '90%',
                    'display': 'flex',
                    'flex-direction': 'row',
                    'justify-content': 'center'
                }
            )
        ], style={
            'width': '75%',
            'display': 'flex',
            'flex-direction': 'column',
            'align-items': 'center',
            'margin-top': '80px'
        })
    ], style={
        'width': '100%',
        'display': 'flex',
        'flex-direction': 'row',
        'justify-content': 'center'
    }),


], style={
    'width': '100%',
    'display': 'flex',
    'flex-direction': 'column',
    'align-items': 'center'
})


@app.callback(
    Output("graph_2", "figure"),
    Input("fuel_2", "value"),
    Input("start_year_2", "value"),
    Input("end_year_2", "value"),
    Input("name_2", "value")
)
def update_chart(fuel_type, start_year, end_year, brand):
    mask = (df['Fuel_Type'] == fuel_type) if fuel_type != 'All' else np.full(len(df), True)
    mask &= (df['Year'].between(start_year, end_year)) if start_year <= end_year else np.full(len(df), False)
    mask &= (df['Name'] == brand) if brand != 'All' else np.full(len(df), True)

    df_filtered = df[mask]
    df_filtered['Price_EUR'] = df_filtered['Price_Rupie'] / conversion_rate
    df_filtered['Price_EUR'] = df_filtered['Price_EUR'].round(2)

    avg_price_by_year = df_filtered.groupby('Year')['Price_EUR'].mean().reset_index()

    fig = px.line(
        avg_price_by_year, 
        x='Year', 
        y='Price_EUR',
        title='Évolution des prix par type de carburant, année et nom',
        labels={"Year": "Années", "Price_EUR": "Prix"}
    )

    fig.update_layout(
        title=dict(
            font=dict(
                family='Arial',
                size=24,
                color='brown'
            ),
            x=0.5,
            xanchor='center'
        )
    )

    return fig


@app.callback(
    Output("graph_1", "figure"),
    Input("fuel_1", "value"),
    Input("start_year_1", "value"),
    Input("end_year_1", "value"),
    # Input("name_1", "value")
)
def update_chart(fuel_type, start_year, end_year):
    mask = (df["Fuel_Type"] == fuel_type) if fuel_type != "All" else df["Fuel_Type"].isin(df["Fuel_Type"].unique())
    mask &= (df["Year"] >= start_year) & (df["Year"] <= end_year)
    # mask &= (df["Name"] == brand) if brand != "All" else df["Name"].isin(df["Name"].unique())

    df_filtered = df[mask]
    df_filtered['Price_EUR'] = df_filtered['Price_Rupie'] / conversion_rate
    df_filtered['Price_EUR'] = df_filtered['Price_EUR'].round(2)

    price_change = df_filtered.groupby(["Year", "Name"])['Price_EUR'].mean().reset_index()
    price_change = price_change.rename(columns={"Price_EUR": "Price_Change"})
    price_change = price_change[~price_change["Price_Change"].isnull()]

    max_price_change = price_change.loc[price_change.groupby("Name")["Price_Change"].idxmax()]

    fig = px.bar(
        max_price_change, 
        x="Name", 
        y="Price_Change", 
        title="Plus grande évolution des prix",
        labels={"Name": "Marques", "Price_Change": "Prix"}
    )

    fig.update_layout(
        title=dict(
            font=dict(
                family='Arial',
                size=24,
                color='brown'
            ),
            x=0.5,
            xanchor='center'
        )
    )

    return fig



if __name__ == "__main__":
    app.run_server(debug=True, port=8050)


