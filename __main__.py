from logic import GoogleMapFetch as gmf
from logic import Anomaly as A
import dash
from dash.dependencies import Event, Output, Input
import dash_html_components as html
import dash_core_components as dcc
import plotly
import base64
from skimage import io
from collections import deque
import plotly.graph_objs as go
from PIL import Image
import numpy as np
import os

cwd = os.getcwd()
#relative to visual studio code folder
datafolder = cwd+"/data/"
evalfolder = cwd+"/eval/"
modelfolder = cwd+"/model/"


my_google_api_key = "" #fill in your own api key

resX = 400
resY = 400
anomaly = A.Anomaly(resX,resY, 48, color=False)
I = 0
#graph data
X = deque(maxlen=20)
X.append(1)
Y = deque(maxlen=20)
Y.append(1)

anomaly.loadModel(modelfolder+'model.h5')
#For model visualization
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


app = dash.Dash()
encoded_images = []
images = []
for image_filename in os.listdir(evalfolder):
    images.append(evalfolder+image_filename)
    encoded_images.append(base64.b64encode(open(evalfolder+image_filename, 'rb').read()))

images_A = [] #np array
images_B = anomaly.eval(evalfolder) #np array
for image in images:  
    #prepare for mse
    image_A = io.imread(images[I] , as_grey=True)
    image_A = image_A.reshape(resX,resY,1)
    image_A = np.array(image_A).astype('float32')
    images_A.append(image_A) 

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
    dcc.Graph(id='live-graph', animate=True),
    html.H2('Original Image & Attempted Regeneration:' , style={'display': 'inline-block'}),  
    html.Div(style={'display': 'inline-block'}),       
    html.H4(''),
    html.Div(
        html.Img(id='original-img',src='data:image/png;base64,{}'.format(encoded_images[0].decode())
        ), style={'display': 'inline-block'}),      
    html.Div(
        html.Img(id='predicted-img',src='data:image/png;base64,{}'.format(encoded_images[1].decode())
        ),style={'display': 'inline-block'}),
    html.H4('Test',style={'display': 'inline-block'}),       
    dcc.Interval(
        id='interval-component',
        interval=5*1000, # in milliseconds
        n_intervals=0
    )
], style={'width': '100%', 'display': 'inline-block'})

@app.callback(Output('original-img', 'src'),
              [Input('interval-component', 'n_intervals')])
def update_original(n):
    global I
    I += 1
    print('current iteration of originals: {0}'.format(I-1))
    return 'data:image/png;base64,{}'.format(encoded_images[I-1].decode())

@app.callback(Output('predicted-img', 'src'),
              [Input('interval-component', 'n_intervals')])
def update_predicted(n):
    temp_img = Image.fromarray(images_B[I], 'RGB')
    temp_img.save(cwd+'temp.png')
    e_img = base64.b64encode(open(cwd+'temp.png', 'rb').read())

    #return reconstruction
    return 'data:image/png;base64,{}'.format(e_img.decode())


@app.callback(Output('live-graph', 'figure'),
              events=[Event('interval-component', 'interval')])
def update_graph():
    #add new mse to graph and
    mse = anomaly.mse(images_A[I], images_B[I]) 
    X.append(X[-1]+1)
    Y.append(mse)
    data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode= 'lines+markers'
            )        
    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                yaxis=dict(range=[min(Y),max(Y)]),)}


def loadModel():
    anomaly.loadModel(modelfolder+'model.h5')
    return

def evaluate():
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
            loadModel()
            evaluate()
        elif arg == 'd':
            getTrainingData()
        elif arg == 'w':
            break
        else: 
            print('Invalid input!')

    app.run_server(debug=False)
    return




if __name__ == "__main__":
    main()
    