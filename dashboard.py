import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
import dash
from dash import dcc, html
import dash_table

# Load Data
crime_data = pd.read_csv("crime_dataset_india.csv")

# Preprocessing
crime_data['Date Reported'] = pd.to_datetime(crime_data['Date Reported'], errors='coerce')
crime_data['Date of Occurrence'] = pd.to_datetime(crime_data['Date of Occurrence'], errors='coerce')
crime_data['Time of Occurrence'] = pd.to_datetime(crime_data['Time of Occurrence'], format='%d-%m-%Y %H:%M', errors='coerce')

# Extract Features
crime_data['Year'] = crime_data['Date of Occurrence'].dt.year
crime_data['Month'] = crime_data['Date of Occurrence'].dt.month
crime_data['Day'] = crime_data['Date of Occurrence'].dt.day
crime_data['Hour'] = crime_data['Time of Occurrence'].dt.hour

# Encoding 'Case Closed' column
crime_data_encoded = pd.get_dummies(
    crime_data, 
    columns=['City', 'Crime Description', 'Crime Domain', 'Case Closed'], 
    drop_first=True
)

# Define target and feature variables
y = crime_data_encoded['Case Closed_Yes']  # Adjust based on encoding
X = crime_data_encoded[['Year', 'Month', 'Day', 'Hour'] + [col for col in crime_data_encoded.columns if col.startswith('City_')]]

# Train Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Time Series Forecasting (ARIMA)
crime_counts = crime_data.set_index('Date of Occurrence').resample('M').size()
arima_model = ARIMA(crime_counts, order=(5, 1, 0))
arima_result = arima_model.fit()
forecast = arima_result.forecast(steps=12)

# Dash App Setup
app = dash.Dash(__name__)

# City-wise Crime Analysis
city_monthly_crimes = crime_data.groupby(['City', 'Year', 'Month']).size().reset_index(name='Crime Count')
city_features = city_monthly_crimes[['Crime Count']]
kmeans = KMeans(n_clusters=5, random_state=42)
city_monthly_crimes['Cluster'] = kmeans.fit_predict(city_features)

# Layout for Dash
app.layout = html.Div([
    html.Nav([
        html.Div([
            html.A("Home", href="#home", id="home-link", style={'padding': '15px', 'color': '#fff', 'cursor': 'pointer'}),
            html.A("Crime Clusters", href="#clusters", id="clusters-link", style={'padding': '15px', 'color': '#fff', 'cursor': 'pointer'}),
            html.A("Crime Trends", href="#trends", id="trends-link", style={'padding': '15px', 'color': '#fff', 'cursor': 'pointer'}),
            html.A("Predictions", href="#predictions", id="predictions-link", style={'padding': '15px', 'color': '#fff', 'cursor': 'pointer'}),
        ], style={'backgroundColor': '#2C3E50', 'display': 'flex'})
    ]),

    html.Div(id="home", children=[
    html.H1(
        "CrimeRadar - Crime Hotspot Prediction Dashboard",
        style={
            'textAlign': 'center',
            'color': '#2980B9',  # A vibrant color for the text
            'fontSize': '40px',  # Larger font size
            'fontWeight': 'bold',  # Bold text
            'padding': '20px',  # Add padding around the text for spacing
            'textDecoration': 'underline',  # Underline the text
            'textTransform': 'uppercase',  # Make text uppercase for emphasis
            'backgroundColor': '#f0f3f5',  # Soft background color behind the title
            'borderRadius': '10px',  # Rounded corners around the background
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',  # Soft shadow effect
            'width': '80%',  # Set width of the title to make it more prominent
            'margin': '0 auto',  # Center align it
        }
    ),
]),


    # Sidebar
    html.Div([
        html.H4("Select Options", style={'fontSize': 20}),
        
        # Dropdown for City Selection
        html.Label("Select City", style={'fontSize': 15}),
        dcc.Dropdown(
            id='city-dropdown',
            options=[{'label': city, 'value': city} for city in crime_data['City'].unique()],
            value=crime_data['City'].unique()[0],
            style={'width': '100%'}
        ),
        
        # Slider for Month Selection
        html.Label("Select Month", style={'fontSize': 15}),
        dcc.Slider(
            id='month-slider',
            min=1,
            max=12,
            value=1,
            marks={i: str(i) for i in range(1, 13)},
            step=1,
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        
        # Slider for Hour Selection
        html.Label("Select Hour of Day", style={'fontSize': 15}),
        dcc.Slider(
            id='hour-slider',
            min=0,
            max=23,
            value=12,
            marks={i: str(i) for i in range(0, 24, 2)},
            step=1,
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        
        # Dropdown for Crime Type Prediction
        html.Label("Select Crime Type", style={'fontSize': 15}),
        dcc.Dropdown(
            id='crime-type-dropdown',
            options=[{'label': crime, 'value': crime} for crime in crime_data['Crime Description'].unique()],
            value=crime_data['Crime Description'].unique()[0],
            style={'width': '100%'}
        ),

    ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),

    # Prediction Table
    html.Div([
        html.H3("Predictions", style={'color': '#2980B9'}),
        dash_table.DataTable(
            id='prediction-table',
            columns=[
                {'name': 'Prediction Type', 'id': 'prediction_type'},
                {'name': 'Prediction Result', 'id': 'prediction_result'}
            ],
            style_table={'height': '400px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': 15},
            style_header={'backgroundColor': '#2C3E50', 'color': 'white'},
            style_data={'backgroundColor': '#F5F5F5', 'color': '#2C3E50'},
        )
    ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px'}),

    # City-wise Clusters
    html.Div(id="clusters", children=[
        html.H3("City-wise Crime Clusters", style={'color': '#2980B9'}),
        dcc.Graph(id='cluster-graph'),
    ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px'}),

    # Crime Trends Plot
    html.Div(id="trends", children=[
        html.H3("Crime Trends Over Time", style={'color': '#2980B9'}),
        dcc.Graph(id='crime-trends-graph'),
    ], style={'width': '100%', 'padding': '20px'}),

    # Crime Distribution Pie Chart for Selected City
    html.Div(id="city-crime-distribution", children=[
        html.H3("Crime Distribution by City", style={'color': '#2980B9'}),
        dcc.Graph(id='city-crime-pie-chart'),
    ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px'}),

    # Crime Type Distribution Pie Chart for Selected Crime
    html.Div(id="crime-type-distribution", children=[
        html.H3("Crime Type Distribution", style={'color': '#2980B9'}),
        dcc.Graph(id='crime-type-pie-chart'),
    ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px'}),

    # Footer
    html.Footer([
        html.Div("© 2025 PitchX. All Rights Reserved.", style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#2C3E50', 'color': 'white'})
    ], style={'width': '100%',  'bottom': '0', 'backgroundColor': '#2C3E50'})
])

# Callbacks to update graphs and predictions
@app.callback(
    [dash.dependencies.Output('cluster-graph', 'figure'),
     dash.dependencies.Output('crime-trends-graph', 'figure'),
     dash.dependencies.Output('prediction-table', 'data'),
     dash.dependencies.Output('city-crime-pie-chart', 'figure'),
     dash.dependencies.Output('crime-type-pie-chart', 'figure')],
    [dash.dependencies.Input('city-dropdown', 'value'),
     dash.dependencies.Input('month-slider', 'value'),
     dash.dependencies.Input('hour-slider', 'value'),
     dash.dependencies.Input('crime-type-dropdown', 'value')]
)
def update_dashboard(selected_city, selected_month, selected_hour, selected_crime_type):
    # Filter the crime data based on selected city
    city_filtered_data = crime_data[crime_data['City'] == selected_city]

    # Cluster Visualization
    city_monthly_crimes_filtered = city_monthly_crimes[city_monthly_crimes['City'] == selected_city]
    fig1 = px.scatter(city_monthly_crimes_filtered, x='Month', y='Crime Count', color='Cluster', title=f"Crime Clusters in {selected_city}")

    # Crime Trend Visualization (Monthly)
    fig2 = px.line(crime_counts, title="Crime Trend (Next Month)")

    # Crime Likelihood Prediction
    pred_data = pd.DataFrame({'Year': [2025], 'Month': [selected_month], 'Hour': [selected_hour]})
    pred_data = pd.get_dummies(pred_data).reindex(columns=X.columns, fill_value=0)
    prediction = rf_model.predict(pred_data)
    
    crime_likelihood = "Crime Likely" if prediction[0] else "No Crime"
    crime_trend = f"Crime Trend for next month: {forecast[0]}"
    hotspot_prediction = f"Hotspot Alert: {selected_city} in Month {selected_month}."
    cluster_risk = f"Cluster Risk: {city_monthly_crimes_filtered['Cluster'].iloc[0]}"
    seasonal_prediction = "Peak Crime Season Detected!" if selected_month in [5, 6, 7, 8] else ""
    future_crime_probability = f"Future Crime Probability for Hour {selected_hour}: 75%"  # Example
    specific_crime_prediction = f"Predicted Crime Type: {selected_crime_type}"  # Example

    # Prepare Prediction Table Data
    prediction_data = [
        {'prediction_type': 'Crime Likelihood', 'prediction_result': crime_likelihood},
        {'prediction_type': 'Crime Trend (Next Month)', 'prediction_result': crime_trend},
        {'prediction_type': 'Hotspot Prediction', 'prediction_result': hotspot_prediction},
        {'prediction_type': 'Cluster Risk', 'prediction_result': cluster_risk},
        {'prediction_type': 'Seasonal Crime Prediction', 'prediction_result': seasonal_prediction},
        {'prediction_type': 'Future Crime Probability', 'prediction_result': future_crime_probability},
        {'prediction_type': 'Predicted Crime Type', 'prediction_result': specific_crime_prediction}
    ]

    # City-wise Crime Distribution Pie Chart
    city_crime_count = city_filtered_data['Crime Description'].value_counts()
    city_pie_chart = {
        'data': [go.Pie(labels=city_crime_count.index, values=city_crime_count.values, hole=0.4)],
        'layout': go.Layout(title=f"Crime Distribution in {selected_city}", showlegend=True)
    }

    # Crime Type Distribution Pie Chart
    crime_type_count = crime_data[crime_data['Crime Description'] == selected_crime_type]['City'].value_counts()
    crime_type_pie_chart = {
        'data': [go.Pie(labels=crime_type_count.index, values=crime_type_count.values, hole=0.4)],
        'layout': go.Layout(title=f"City Distribution for {selected_crime_type} Crimes", showlegend=True)
    }

    return fig1, fig2, prediction_data, city_pie_chart, crime_type_pie_chart

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
