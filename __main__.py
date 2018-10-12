from logic import GoogleMapFetch as gmf
from logic import Anomaly as A
import dash
from dash.dependencies import Event, Output
import dash_html_components as html
import dash_core_components as dcc
import base64
import os

cwd = os.getcwd()
#relative to visual studio code folder
datafolder = cwd+"/data/"
evalfolder = cwd+"/eval/"
modelfolder = cwd+"/model/"


my_google_api_key = "" #fill in your own api key

#For model visualization
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


app = dash.Dash()
encoded_images = []
for image_filename in os.listdir(evalfolder):
    encoded_images.append(base64.b64encode(open(evalfolder+image_filename, 'rb').read()))

app.layout = html.Div(children=[
    html.H1(children='Geospatial Anomaly Detection'),
    dcc.Markdown('''
    #### Dash and Markdown

    Dash supports [Markdown](http://commonmark.org/help).

    Markdown is a simple way to write and format text.
    It includes a syntax for things like **bold text** and *italics*,
    [links](http://commonmark.org/help), inline `code` snippets, lists,
    quotes, and more.
    '''),
    html.Div(
        dcc.Graph(id='my-graph',        
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }}
        ), style={'marginBottom': 50, 'marginTop': 25}),
    html.Div(
        html.Img(src='data:image/png;base64,{}'.format(encoded_images[0].decode())
        ), style={'display': 'inline-block'}),
    html.Div(
        html.Img(src='data:image/png;base64,{}'.format(encoded_images[1].decode())
        ),style={'display': 'inline-block'}),
    html.Div(
        html.Img(src='data:image/png;base64,{}'.format(encoded_images[2].decode())
        ), style={'display': 'inline-block'}),
    html.Div(
        html.Img(src='data:image/png;base64,{}'.format(encoded_images[3].decode())
        ),style={'display': 'inline-block'})
], style={'width': '100%', 'display': 'inline-block'})



resX = 400
resY = 400
anomaly = A.Anomaly(resX,resY, 48, color=False)

def evaluate():
    anomaly.loadModel(modelfolder+'model.h5')
    #evaldata = gmf.GoogleMapFetch(400, 33.724084, 51.722354, zoom=18, apikey=my_google_api_key)
    #evaldata.generateImages(evalfolder, 2, 2)
    anomaly.eval(evalfolder)
    return

def getTrainingData():
    # Create a new instance of GoogleMap Downloader
    # default 34.872548, 51.427510 
    trainingdata = gmf.GoogleMapFetch(400, 34.872548, 51.427510, zoom=18, apikey=my_google_api_key)
    trainingdata.generateImages(datafolder, 80, 80)
    print("------The satellite imagery for training has successfully been created!------\n")
    return

def trainModel():
    anomaly.loadTrainingData(datafolder, testpercent=10)
    anomaly.createModel()
    anomaly.train(modelfolder)
    return

def main():

    print("Automated Geospatial Anomaly Detection \nV1.0 - Taylor McNally\n")

    while True:
        arg = input('Type the character corresponding to the desired task: t for training, e for evaluation, or d for fetching training data, w for launching dash server.')
        if arg == 't':
            trainModel()
        elif arg == 'e':
            evaluate()
        elif arg == 'd':
            getTrainingData()
        elif arg == 'w':
            app.run_server(debug=True)
        else: 
            print('Invalid input!')

    return




if __name__ == "__main__":
    main()
    