import GoogleMapFetch as gmf
import Anomaly as A
import os

#relative to visual studio code folder
cwd = os.getcwd()
datafolder = cwd + "/data/"
evalfolder = cwd + "/eval/"
modelfolder = cwd + "/model/"

my_google_api_key = "" #fill in your own api key

#For model visualization
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

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
    anomaly = A.Anomaly(resX,resY, 48, color=True)
    
    anomaly.loadTrainingData(datafolder, testpercent=10)
    anomaly.createModel()
    anomaly.train(modelfolder)

    anomaly.loadModel(modelfolder+'model.h5')
    
    print("------Anomaly detection ready!------\n")

    evaldata = gmf.GoogleMapFetch(400, 33.724084, 51.722354, zoom=18, apikey=my_google_api_key)
    evaldata.generateImages(evalfolder, 2, 2)
    anomaly.eval(evalfolder)


if __name__ == '__main__':  main()