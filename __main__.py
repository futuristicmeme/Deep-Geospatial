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


# app = dash.Dash()
# image_filename = evalfolder+'0-0.png' # replace with your own image
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),
#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),
#     html.Div(
#         dcc.Graph(id='my-graph'
#         ), style={'display': 'inline-block'}),
#     html.Div(
#         dcc.Graph(id='my-graph2'
#         ), style={'display': 'inline-block'}),
#     html.Div(
#         html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())
#         ),style={'display': 'inline-block'})
# ], style={'width': '100%', 'display': 'inline-block'})


# app.run_server(debug=True)

def main():

    print("Automated Geospatial Anomaly Detection \nV1.0 - Taylor McNally\n")

    # Create a new instance of GoogleMap Downloader
    # default 34.872548, 51.427510 
    trainingdata = gmf.GoogleMapFetch(400, 34.872548, 51.427510, zoom=18, apikey=my_google_api_key)

    # Get the high resolution satellite images
    trainingdata.generateImages(datafolder, 80, 80)

    print("------The satellite imagery for training has successfully been created!------\n")
    
    resX = 400
    resY = 400
    anomaly = A.Anomaly(resX,resY, 48, color=False)
    
    anomaly.loadTrainingData(datafolder, testpercent=10)
    anomaly.createModel()
    anomaly.train(modelfolder)

    anomaly.loadModel(modelfolder+'model.h5')
    
    print("------Anomaly detection ready!------\n")

    evaldata = gmf.GoogleMapFetch(400, 33.724084, 51.722354, zoom=18, apikey=my_google_api_key)
    evaldata.generateImages(evalfolder, 2, 2)
    anomaly.eval(evalfolder)


if __name__ == '__main__':  main()
    