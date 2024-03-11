
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import flask
import pandas as pd
import plotly.express as px
import plotly.utils as utils
import json

app = dash.Dash(__name__)
server = app.server

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

@server.route('/login')
def login():
    graph_json = json.dumps(fig, cls=utils.PlotlyJSONEncoder)
    return flask.render_template('login.html', graph=graph_json)

@server.route('/register')
def register():
    return flask.render_template('register.html')

if __name__ == '__main__':
    app.run_server(debug=True)
