from vis.visualization import visualize_activation
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
from keras import applications
import matplotlib.pyplot as plt

from PIL import Image
import requests
import sys
from io import BytesIO
import matplotlib.pyplot as plt
#importing required libraries and functions
from keras.models import Model
from keras.preprocessing.image import img_to_array

from keras.applications import VGG16
model = VGG16(weights='imagenet',include_top=True)
#from keras.preprocessing import image
#from keras.preprocessing.image import reshape
#defining names of layers from which we will take the output
layer_names = ['block1_conv1','block2_conv1','block3_conv1','block4_conv2']
outputs = []

#url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/250px-Gatto_europeo4.jpg'
#response = requests.get(url)
image1=sys.argv[1]
image = Image.open(image1)
image = image.crop((0, 0, 224, 224))
image = img_to_array(image)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#extracting the output and appending to outputs
for layer_name in layer_names:
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(image)
    outputs.append(intermediate_output)
#plotting the outputs
fig,ax = plt.subplots(nrows=4,ncols=5,figsize=(20,20))

for i in range(4):
    for z in range(5):
        ax[i][z].imshow(outputs[i][0,:,:,z])
        ax[i][z].set_title(layer_names[i])
        ax[i][z].set_xticks([])
        ax[i][z].set_yticks([])
plt.savefig('layerwise_output.jpg')


layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

#generating saliency map with unguided backprop
grads1 = visualize_saliency(model, layer_idx,filter_indices=None,seed_input=image)
grads2 = visualize_saliency(model, layer_idx,filter_indices=None,seed_input=image,backprop_modifier='guided')
plt.imsave('unguided_saliency.jpg', grads1)
plt.imsave('guided_saliency.jpg', grads2)