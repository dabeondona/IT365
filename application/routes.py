from application import app
from flask import render_template, url_for
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from numpy import math
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objs as go
import geopandas as gpd
import pandas as pd

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/plots')
def plot():
    
    df = pd.read_csv("kc_house_data_third.csv")
    df['price'] = df['price'].apply(lambda x: int(x))
    
    X = df[['sqft_above', 'sqft_basement']]
    y = df['price']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    z_scores = np.abs(stats.zscore(X_train))
    threshold = 3 
    outlier_indices = np.where(z_scores > threshold)

    X_train_no_outliers = X_train[(z_scores <= threshold).all(axis=1)]
    y_train_no_outliers = y_train[(z_scores <= threshold).all(axis=1)]
    
    model = LinearRegression()
    model.fit(X_train_no_outliers, y_train_no_outliers)
    y_pred = model.predict(X_test)
        
    fig1 = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Prices', 'y': 'Predicted Prices'})
    fig1.add_shape(type='line', x0=min(y_test), x1=max(y_test), y0=min(y_test), y1=max(y_test), line=dict(color='red', dash='dash'), name='Perfect Fit')
    fig1.update_layout(
        title='Actual vs. Predicted Prices',
        template="plotly_dark"
    )
    
    a1 = r2_score(y_test, y_pred)
    b1 = model.intercept_
    c1 = math.sqrt(mean_squared_error(y_test, y_pred))
   
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

# /-----------------------------------------------------------------------------------------------------------------------------------------------------------

    df = px.data.iris()
    fig2 = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species',  title="Iris Dataset", template="plotly_dark")
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

# /-----------------------------------------------------------------------------------------------------------------------------------------------------------


    df = px.data.gapminder().query("continent=='Oceania'")
    fig3 = px.line(df, x="year", y="lifeExp", color='country',  title="Life Expectancy", template="plotly_dark")
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('index.html', graph1JSON=graph1JSON, a1=a1, b1=b1, c1=c1, graph2JSON=graph2JSON, graph3JSON=graph3JSON, title="Plots", )

@app.route('/layout')
def layout():
    return render_template('layout.html', title="Layout")

@app.route('/mapplot')
def mapplot():
    world_geojson = gpd.read_file('countries.geojson')

    url = "https://raw.githubusercontent.com/amin0930/YouTube-Statistics/master/Global%20YouTube%20Statistics.csv"
    youtuber_data = pd.read_csv(url, encoding='ISO-8859-1')
    youtuber_data_cleaned = youtuber_data.dropna(subset=['Country'])

    country_counts = youtuber_data_cleaned['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Number']
    world_geojson['ADMIN'].replace('United States of America', 'United States', inplace=True)
    merged_data = world_geojson.merge(country_counts, left_on='ADMIN', right_on='Country', how='left')
    merged_data.fillna(0, inplace=True)
    
    country_dataframe = merged_data[['Country', 'Number']]
    country_dataframe['Number'] = country_dataframe['Number'].astype(int)
    country_dataframe = country_dataframe.query('Country != "0" and Number != 0')
    country_dataframe = country_dataframe.sort_values(by='Number', ascending=False)
    
    fig = px.choropleth(merged_data, geojson=merged_data.geometry, locations=merged_data.index, color='Number', hover_name='ADMIN', projection='mercator', title='Amount of YouTubers per Country', color_continuous_scale='YlOrRd')
    fig.update_geos(showcoastlines=True, coastlinecolor="Gray", showland=True, landcolor="white")
    fig.update_layout(height=1000, width=1000)
    fig.update_layout(
    paper_bgcolor='rgb(29, 29, 29)',
    plot_bgcolor='rgb(29, 29, 29)',
    font=dict(color='white'))
    map_json = fig.to_json()
    return render_template('mapplot.html', map_json=map_json, country_dataframe=country_dataframe, title="Plot Map")
