import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    dcc.Graph(id='stock-graph'),
    dcc.Interval(id='interval-component', interval=10*1000, n_intervals=0),
    dcc.Dropdown(
        id='stock-dropdown',
        options=[
            {'label': 'Apple', 'value': 'AAPL'},
            {'label': 'Google', 'value': 'GOOGL'},
            {'label': 'Amazon', 'value': 'AMZN'}
        ],
        value='AAPL',  # Default value
        multi=False
    )
])

@app.callback(
    Output('stock-graph', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('stock-dropdown', 'value')
)
def update_graph(n, selected_stock):
    # Fetch historical data for the selected stock
    df = yf.download(selected_stock, period='1d', interval='1m')
    if df.empty:
        return {}

    # Create the figure
    figure = {
        'data': [
            {
                'x': df.index,
                'y': df['Close'],
                'type': 'line',
                'name': selected_stock
            }
        ],
        'layout': {
            'title': f'Real-Time Stock Data for {selected_stock}',
            'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'Price (USD)'}
        }
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
