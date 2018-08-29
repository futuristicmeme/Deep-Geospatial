import urllib.request
from PIL import Image
import os
import math


class GoogleMapFetch:
    """
        A class which generates high resolution Google Maps images given
        a longitude, latitude and zoom level
    """

    def __init__(self, res, lat, lng, zoom=12, type='satellite', apikey=''):
        """
            GoogleMapFetch Constructor
            Args:
                res:        The resolution of the image
                lat:        The latitude of the location required
                lng:        The longitude of the location required
                zoom:       The zoom level of the location required, ranges from 0 - 23
                            defaults to 12
                type:       The type of map acquired ex: roadmap, satellite, terrain, hybrid 
                            defaults to satellite     
                apikey:     Your Google Maps api goes here             
        """
        self.res = res
        self._lat = lat
        self._lng = lng
        self._zoom = zoom
        self._type = type
        self.apikey = apikey

    
    def generateImage(self, start_x=None, start_y=None):
        """
            Generates a Google Maps image.

            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate

            Returns:
                A Google Map image.
        """

        if start_x == None or start_y == None:
            start_x, start_y = self._lat, self._lng


        url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + str(start_x) + ',' + str(start_y) + '&zoom=' + str(
            self._zoom)+ '&size='+ str(self.res) + 'x' + str(self.res)+ '&maptype='+ str(self._type) + '&key=' + str(self.apikey) 

        current_tile = 'temp'
        urllib.request.urlretrieve(url, current_tile)
        im = Image.open(current_tile)

        return im

    def generateImages(self, path, imgs_x, imgs_y, start_x=None, start_y=None):
        """
            Generates images by fetching a number of Google Map tiles.

            Args:
                path:           The path to save the tiles
                imgs_x:         The number of images on the x-axis
                imgs_y:         The number of images on the y-axis
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
        """

        if start_x == None or start_y == None:
            start_x, start_y = self._lat, self._lng

        for x in range(0,imgs_x):
            for y in range(0, imgs_y):
                img = self.generateImage(start_x=start_x+x*(0.001),start_y=start_y+y*(0.001)) # hard coded for now
                #img.show()
                savepath = path+str(x)+"-"+str(y)+".png"
                img.save(savepath)
                #os.remove(str(x) + '-' + str(y))

        return
