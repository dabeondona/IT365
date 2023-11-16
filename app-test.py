import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import geopandas as gpd
import pandas as pd
import plotly.express as px

app = dash.Dash(__name__, title="Hmm")
world_geojson = gpd.read_file('countries.geojson')

url = "https://raw.githubusercontent.com/amin0930/YouTube-Statistics/master/Global%20YouTube%20Statistics.csv"
youtube_data = pd.read_csv(url, encoding='ISO-8859-1')
youtube_data_cleaned = youtube_data.dropna(subset=['Country'])

country_counts = youtube_data_cleaned['Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Number']
world_geojson['ADMIN'].replace('United States of America', 'United States', inplace=True)
merged_data = world_geojson.merge(country_counts, left_on='ADMIN', right_on='Country', how='left')
merged_data.fillna(0, inplace=True)

initial_map = px.scatter_geo(
    merged_data,
    locations='ADMIN', 
    locationmode='country names',
    color='Number',
    hover_name='ADMIN',  
    size='Number',
    projection='mercator',
    height=750,
    width=750,
    title='YouTube Statistics by Country'
)

initial_map.update_geos(
    showcoastlines=True, 
    coastlinecolor="Gray", 
    showcountries=True,  
    countrycolor="Black" 
)

app.layout = html.Div([
    dcc.Graph(id='world-map',
              config={'scrollZoom': False},
              figure=initial_map), 
    html.Div(id='youtubers-list') 
])

@app.callback(
    Output('world-map', 'figure'),
    Output('youtubers-list', 'children'), 
    [Input('world-map', 'clickData')] 
)

def update_map(click_data):

    if click_data is not None:
        selected_country = click_data['points'][0]['hovertext']

        youtubers_for_country = youtube_data_cleaned[youtube_data_cleaned['Country'] == selected_country][['Youtuber', 'rank', 'subscribers']]
        
        youtubers_for_country = youtubers_for_country.sort_values(by='rank')

        table = dash_table.DataTable(
            columns=[
                {'name': 'Youtuber', 'id': 'Youtuber'},
                {'name': 'rank', 'id': 'rank'},
                {'name': 'subscribers', 'id': 'subscribers'},
            ],
            data=youtubers_for_country.to_dict('records'),
            style_table={'overflowX': 'auto'},
        )

        fig = px.scatter_geo(
            merged_data,
            locations='ADMIN', 
            locationmode='country names',
            color='Number',
            hover_name='ADMIN',  
            size='Number',
            projection='mercator',
            title=f'YouTube Statistics by Country ({selected_country})'
        )
        
        fig.update_geos(
            showcoastlines=True, 
            coastlinecolor="Gray", 
            showcountries=True,  
            countrycolor="Black" 
        )

        return fig, table

    return initial_map, []

if __name__ == '__main__':
    app.run_server(debug=True)
