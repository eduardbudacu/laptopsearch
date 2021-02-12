import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA 
import numpy as np

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

import sys

DATE = sys.argv[1]

itgalaxy = pd.read_csv('data/' + DATE + '/itgalaxy.csv')
itgalaxy['price'] = itgalaxy['price'].apply(lambda x: float(x.replace('.', '').replace(',','.').replace(' lei', '')))
itgalaxy['provider'] = 'itgalaxy'

emag = pd.read_csv('data/' + DATE + '/emag.csv')
emag['price'] = emag['price'].apply(lambda x: float(x.replace('<span class="font-size-sm">de la</span> ', '').replace('.', '').replace('<sup>', '.').replace('</sup> <span>Lei</span>', '')))
emag['provider'] = 'emag'
emag['title'] = emag['title'].apply(lambda x: x.replace('®', '').replace('™', '').replace(',', ''))

evomag = pd.read_csv('data/' + DATE + '/evomag.csv')
evomag['price'] = evomag['price'].apply(lambda x: float(x.replace('.', '').replace(',', '.').replace(' Lei', '')))
evomag['provider'] = 'evomag'

pcgarage = pd.read_csv('data/' + DATE + '/pcgarage.csv')
pcgarage['price'] = pcgarage['price'].apply(lambda x: float(x.replace('.', '').replace(',', '.').replace(' RON', '')))
pcgarage['provider'] = 'pcgarage'
pcgarage['image'] = pcgarage['image'].apply(lambda x: x.split(',')[0])
pcgarage = pcgarage.rename(columns={'image':'image-src'})

flanco = pd.read_csv('data/' + DATE + '/flanco.csv')

flanco['price'] = flanco['price'].apply(lambda x: float(str(x).replace('.', '').replace(',', '.').replace(' lei', '')))
flanco['provider'] = 'flanco'
mediagalaxy = pd.read_csv('data/' + DATE + '/mediagalaxy.csv')
mediagalaxy['price'] = mediagalaxy['price'].apply(lambda x: float(str(x).replace('.', '').replace(',', '.').replace('lei', '')))
mediagalaxy['provider'] = 'mediagalaxy'


all_data = pd.concat([itgalaxy, emag, evomag, pcgarage, flanco, mediagalaxy], ignore_index=True)



all_data = all_data[['title', 'price', 'image-src', 'link-href', 'provider', 'availability']]
availability = {'Stoc epuizat':'indisponibil', 'In Stoc':'disponibil', 'In anumite magazine':'disponibil',  'In stoc - exclusiv online':'disponibil', 'In stoc':'disponibil', 'Stoc limitat':'disponibil','Stoc magazin limitat': 'disponibil', 'Nu este in stoc':'indisponibil', 'Stoc magazin suficient':'disponibil', 'In stoc furnizor':'disponibil', 'Intreaba stoc': 'indisponibil', 'nu este in stoc': 'indisponibil', 'in stoc depozit/furnizor': 'disponibil', 'contactati-ne pentru info stoc': 'indisponibil', 'in stoc limitat': 'disponibil', 'in stoc suficient': 'disponibil', 'în stoc': 'disponibil', 'ultimul produs in stoc': 'disponibil', 'stoc epuizat': 'indisponibil', 'ultimele 3 produse': 'disponibil', 'ultimele 2 produse': 'disponibil', 'indisponibil': 'indisponibil', 'în stoc furnizor': 'disponibil', 'In stoc (exclusiv online)': 'disponibil', 'Stoc epuizat': 'indisponibil', 'Intreaba stoc': 'indisponibil', 'In stoc magazin': 'disponibil', 'Preorder': 'indisponibil'}
all_data['availability'] = all_data['availability'].map(availability)
all_data = all_data.dropna().reset_index()

print(all_data)

def create_model(data):
    extra_stop_words = list(stopwords.words('romanian'))
    extra_stop_words.extend(['cu', 'procesor', 'laptop', 'gaming', 'workstation'])

    stop = text.ENGLISH_STOP_WORDS.union(extra_stop_words)

    vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=80000, analyzer = 'word', stop_words=set(stop))
    vectors = vectorizer.fit_transform(list(data['title']))
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()

    feature_array = np.array(feature_names)
    print(dense.shape)

    #demosntrate a search
    #search = vectorizer.transform(['Legion']).todense()

    #res = np.dot(dense, search[0].transpose())

    #print(res.nonzero()[0])
    #for v in res.nonzero()[0]:
        #print(data.iloc[v].title)
    #print(len(res.nonzero()[0]))
    top_keywords = []
    for i,val in enumerate(vectors):
        sorting = np.argsort(vectors[i].toarray()).flatten()[::-1]
        n = 20
        top_n = feature_array[sorting][:n]
        top_keywords.append(', '.join(top_n))
    

    n_components=2

    pca = PCA(n_components=n_components).fit(dense)
    data2D = pca.transform(dense)

    dataPd = pd.DataFrame.from_records(data2D)
    dataPd.columns = ['x', 'y']

    keywords = pd.DataFrame(top_keywords)
    keywords.columns = ['top_keywords']
    dataPlot = pd.concat([dataPd, data, keywords], axis=1)

    fig = px.scatter(dataPlot, x='x', y='y', custom_data=['title', 'top_keywords', 'price', 'image-src'], color='provider',size='price')
    fig.update_traces(
        hovertemplate="<br>".join([
            "%{customdata[0]}",
            "%{customdata[2]}",
            "%{customdata[1]}"
        ])
    )
    return fig





if __name__ == "__main__":
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    all_data = all_data[all_data['availability'] == 'disponibil'].reset_index(drop=True)

    min_price = all_data['price'].min()
    max_price = all_data['price'].max() 

    print(min_price, max_price)

    fig = create_model(all_data)
    fig.update_layout(clickmode='event+select', height=600)
    app.layout = html.Div(children=[
        html.H1('Laptop search'),
        html.P(children=["Price range [", html.Span(id='min-price', children=[min_price]), ' - ', html.Span(id='max-price', children=[max_price]), ']']),
        dcc.RangeSlider(
            id='range-slider',
            min=min_price, max=max_price, step=100,
            marks={min_price:str(min_price), max_price:str(max_price)},
            value=[min_price, max_price]
        ),
        html.Div(style = {'height': '600px'}, id='plot-container',  children=[dcc.Graph(
        	    id='scatter-plot',
        	    figure=fig
    )]),
                
        html.Div(id='image')
    ])

    @app.callback(
        [Output('scatter-plot', 'figure'), Output('min-price', 'children'), Output('max-price', 'children')],
        [Input('range-slider', 'value')])
    def update_chart(slider_range):
        low, high = slider_range
        mask = (all_data['price'] > low) & (all_data['price'] < high)
        data = all_data[mask]
        fig = create_model(data.reset_index(drop=True))
        return fig,  low, high
    
    @app.callback(
    Output('image', 'children'),
    Input('scatter-plot', 'clickData'))
    def display_click_data(clickData):
        if clickData:
            return [html.H3(clickData['points'][0]['customdata'][0]), html.Img(src="{data}".format(data=clickData['points'][0]['customdata'][3]), width=360)]
        else:
            return "Click to select"
    port = int(os.environ.get("PORT", 8051))
    print(port)
    app.run_server(debug=False, host='0.0.0.0', port=port)
