#importing required modules
import keras
import cv2
from matplotlib import pyplot as plt
from keras.applications import VGG16
#loading the saved model
#we are using the complete architecture thus include_top=True
model = VGG16(weights='imagenet',include_top=True)
#show the summary of model
model.summary()

#creating a mapping of layer name ot layer details 
#we will create a dictionary layers_info which maps a layer name to its charcteristics
layers_info = {}
for i in model.layers:
    layers_info[i.name] = i.get_config()

#here the layer_weights dictionary will map every layer_name to its corresponding weights
layer_weights = {}
for i in model.layers:
    layer_weights[i.name] = i.get_weights()

print(layers_info['block5_conv1'])


layers = model.layers
layer_ids = [1,4,7,11,15]
#plot the filters
fig,ax = plt.subplots(nrows=1,ncols=5)
for i in range(5):
    ax[i].imshow(layers[layer_ids[i]].get_weights()[0][:,:,:,0][:,:,0],cmap='gray')
    ax[i].set_title('block'+str(i+1))
    ax[i].set_xticks([])
    ax[i].set_yticks([])


#importing the required modules
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from keras import applications
import matplotlib.pyplot as plt
#%matpsorflow.lotlib inline
#plt.rcParams['figure.figsize'] = (18,6)
#creating a VGG16 model using fully connected layers also because then we can 
#visualize the patterns for individual category
from keras.applications import VGG16
model = VGG16(weights='imagenet',include_top=True)

#finding out the layer index using layer name
#the find_layer_idx function accepts the model and name of layer as parameters and return the index of respective layer
layer_idx = utils.find_layer_idx(model,'predictions')

print(layer_idx)

#changing the activation of the layer to linear
model.layers[layer_idx].activation = activations.linear
#applying modifications to the model
model = utils.apply_modifications(model)
#Indian elephant
img3 = visualize_activation(model,layer_idx,filter_indices=384,max_iter=5000,verbose=True)
#plt.imshow(img3)
cv2.imwrite('out.jpg',img3)
