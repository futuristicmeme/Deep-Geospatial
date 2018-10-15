from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Dense
from keras.callbacks import TensorBoard
from keras.models import Model, load_model
import keras.backend as K
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from skimage import io
import numpy as np
from sklearn.cross_validation import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import os


errs = ["Model already loaded! Please reset current model to load a new one!",""]

class Anomaly():
    """
        A class which creates an autoencoder designed for detecting anomalies in
        Google Maps imagery
    """
    
    def __init__(self, res_x, res_y, code_dim, color=True):
        """
            Anomaly Constructor
            Args:
                res_x:        The latitude of the location required
                res_y:        The longitude of the location required
                code_dim:     Coded layer dimensions
                color:        Whether a RGB image is used                  
        """
        self.res_x = res_x
        self.res_y = res_y

        self.code_dim = code_dim
        self.model = None
        self.encoder = None
        self.decoder = None

        self.x_train = None
        self.x_test = None

        self.encoded_input = None
        self.decoder_layer = None

        if(color==True):
            self.color=3
        else:
            self.color=1
        
    def loadTrainingData(self, trainpath, testpath=None, testpercent=None, rescaleSize=None):
        """
            Loads and prepares training &/or test data for the model.

            Args:
                trainpath:       The absolute path to the training folder
                testpath:        The absolute path to the testing folder - defaults to None
                testpercent:     The number of tiles wide the image should be -
                                 defaults to None (if testpath is none as well)
                rescaleSize:         Downscale amount. (Ex: 1.0/4.0) If your target resolution is lower than the
                                 network input dimensions use this to rescale your photos. 
        """
        #convert all images in file to keras-readable array
        train_images = []
        test_images = []
        for image_path in os.listdir(trainpath):
            if not image_path.startswith('.'):
                img = io.imread(trainpath+image_path , as_grey=(self.color!=3))
                if(rescaleSize!=None):
                    img = rescale(img, rescaleSize)                
                img = img.reshape(self.res_x,self.res_y,self.color)
                train_images.append(img)

        if(testpercent != None):
            x_all = np.array(train_images).astype('float32')
            length = len(train_images)
            # X% of the data for training,  100-X% of the data for testing
            self.x_train, self.x_test = train_test_split(x_all, test_size=((testpercent*length)//100), random_state=24)
        else:
            for image_path in os.listdir(testpath):
                if not image_path.startswith('.'):
                    img = io.imread(testpath+image_path , as_grey=(self.color!=3))
                    if(rescaleSize!=None):
                        img = rescale(img, rescaleSize)                     
                    img = img.reshape(self.res_x,self.res_y,self.color)
                    test_images.append(img)
            self.x_train = np.array(train_images).astype('float32')
            self.x_test = np.array(test_images).astype('float32')
    
        print(self.x_train.shape)
        print(self.x_test.shape)
        return


    def createModel(self):
        """
            Generates the model structure for use in training and evaluating.

        """

        input_img = Input(shape=(self.res_x,self.res_y,self.color))
        encoded = MaxPooling2D((2, 2), padding='same', input_shape=(self.res_x,self.res_y,self.color))(input_img)
        encoded = Conv2D(int(self.code_dim*2), (3, 3), activation='relu', padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
        encoded = Conv2D(self.code_dim*4, (3, 3), activation='relu', padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)

        decoded = Conv2D(self.code_dim*4, (3, 3), activation='relu', padding='same')(encoded)
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(int(self.code_dim*2), (3, 3), activation='relu', padding='same')(decoded)
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(int(self.code_dim), (3, 3), activation='relu', padding='same')(decoded)
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(self.color, (3, 3), activation='linear', padding='same')(decoded)
        #decoded = LeakyReLU(alpha=.001)

        # maps an input to its reconstruction
        self.model = Model(input_img, decoded)

        # maps an input to its encoded representation
        self.encoder = Model(input_img, encoded)


        self.encoded_input = Input(shape=(self.res_x,self.res_y,self.color))
        self.decoder_layer = self.model.layers[-1] # last layer of the autoencoder model


        self.model.compile(optimizer='adadelta', loss='mean_squared_logarithmic_error')
        plot_model(self.model, to_file="model_plot.png", show_layer_names=True, show_shapes=True)
        return

    def loadModel(self, savePath):
        """
            Loads a saved compiled model.

        """
        if(self.model == None):
            self.model = load_model(savePath)
            return
        print(errs[0])
        return

    def resetModel(self):
        """
            Resets the current model in memory.

        """
        if(self.model != None):
            self.model = None

    def train(self, savePath=None):
        """
            Trains the generated model.
        
            Args:

                (Optional) savePath: location to save model file - defaults to None      

        """
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)
        #tensorboard --logdir path_to_current_dir/Graph to see visual progress
        self.model.fit(self.x_train, self.x_train,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_data=(self.x_test, self.x_test), 
                callbacks=[tbCallBack])
        if(savePath!=None):
            self.model.save(savePath+'model.h5')
        return

    def eval(self, evalimgspath):
        """
            Evaluates the trained model against a set of images.
        
            Args:

                evalimgspath: location of the eval image(s)      

        """
        #compare generated image to original
        eval_images = []
        for image_path in os.listdir(evalimgspath):
            if not image_path.startswith('.'):
                img = io.imread(evalimgspath+image_path , as_grey=(self.color!=3))
                img = img.reshape(self.res_x,self.res_y,self.color)
                eval_images.append(img)
        x_eval = np.array(eval_images).astype('float32')      
        #detect anomalys
        print(x_eval.shape)
        decoded_imgs = self.model.predict(x_eval)

        n = len(eval_images)
        for i in range(n):
            # display original
            # original_img = Image.fromarray(eval_images[i], 'RGB')
            # original_img.show()

            # display reconstruction
            # decoded_img = Image.fromarray(decoded_imgs[i], 'RGB')
            # decoded_img.show()

            print('Mean Squared Error of iteration {0} : {1}'.format(i,self.mse(eval_images[i], decoded_imgs[i])))
        return decoded_imgs

    def evalSingle(self,image_path):
        """
                Evaluates a single image and returns the mse 
            
                Args:

                    evalimgspath: location of the eval image(s)      

        """
                #compare generated image to original
        image_paths = [image_path, image_path]
        eval_images = []
        for image_path in image_paths:    
            img = io.imread(image_path , as_grey=(self.color!=3))
            img = img.reshape(self.res_x,self.res_y,self.color)
            eval_images.append(img)
        x_eval = np.array(eval_images).astype('float32')      
        #detect anomalys
        print(x_eval.shape)
        decoded_imgs = self.model.predict(x_eval)
        print(decoded_imgs.shape)
        return decoded_imgs[0]


    def mse(self, imageA, imageB):
        """
            'Mean Squared Error' between the two images
            Args:
                imageA: numpy matrix
                imageB: numpy matrix

            Returns: 
                Mean squared error
                
        """
        err = np.sum((imageA - imageB) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        

        return err
