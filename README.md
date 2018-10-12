# Deep Geospatial Intelligence

A Keras Python project for detecting anomalies in geospatial maps

### This has more or less just become a playground for testing new stuff out on an existing project... dont expect much readability...

<img src="https://78.media.tumblr.com/d14666bd4de029b7ee6bb61a96d90828/tumblr_inline_ohvef7kB3O1tzhl5u_400.gif" width="400" height="200" />


## News

-August 26, 2018: Initial Commit


## Requirements

* 11 GB VRAM

* Google static maps API

* Anaconda 3.6

* Some neural bois

## Quickstart

Get up and running the pretrained model in seconds

#### Download Google Static Maps

Add your google maps api to the string in the GeospatialIntelligence.py file

> y_google_api_key = "" #fill in your own api key

Go ahead and comment out training the model on line 34ish

> #anomaly.train(modelfolder)

Run the GeospatialIntelligence.py file from the repo's folder

>cd Deep-Geospatial

>python logic/GeospatialIntelligence.py


#### Load Pretrained Model

#### Evaluate Imagery

> mkdir data


<img src="resources/model_plot.png" width="200" height="600" />
