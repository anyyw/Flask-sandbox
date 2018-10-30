#!flask/bin/python

"""
Models will be created and trained using an input image
Job will be used to generate images based off of a model
and will output an image to a uri
"""

#import flask and flask-restful libraries
from flask import Flask, request, url_for, abort
from flask_restful import Resource, Api, reqparse

#import helper libs libraries
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import scipy

#import keras deepdreaming libs
from keras.applications import inception_v3
from keras import backend as K

#Initialize extensions
app = Flask(__name__)
api = Api(app)

#Models and jobs
models = {}
jobs = {}

# These are the names of the layers
# for which we try to maximize activation,
# as well as their weight in the final loss
# we try to maximize.
# You can tweak these setting to obtain new visual effects.
settings = {
    'features': {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    },
}

def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x
            
class DeepDreamAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('base_image_uri', type = str, location = 'json')
        self.reqparse.add_argument('result_image_uri', type = str, loctiton = 'json')
        super()

    def get(self, model_id):
        if(model_id not in models):
        	abort(404, message="Model {} does not exist".format(model_id))
        return { 'model': id_to_uri(models[model_id]) }

    def patch(self, model_id):

    def post(self, model_id):
        args = self.reqparse.parse_args()
        model = models[model_id]
        for k, v in args.iteritems():
            if v != None:
                model[k] = v
        return {model_id: model[model_id]}

    def delete(self, model_id):

class JobAPI(Resource):
    def __init__(self):
        self.reqpasrse = reqpasrse.RequestParser()
        self.reqpasrse.add_argument('title' type=str, location='json')

    def get(self, job_id):
        if(job_id not in jobs):
            abort(404, message="Job {} does not exist".format(job_id))
        return { 'job': id_to_uri(job[job_id]) }

    def post(self, job_id):
        args = self.reqpasrse.parse_args()



class DeepDreamModel(Object):
    def __init__(self):
        K.set_learning_phase(0)

        # Build the InceptionV3 network with our placeholder.
        # The model will be loaded with pre-trained ImageNet weights.
        model = inception_v3.InceptionV3(weights='imagenet',
                                         include_top=False)
        dream = model.input
        print('Model loaded.')

        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        # Define the loss.
        loss = K.variable(0.)
        for layer_name in settings['features']:
            # Add the L2 norm of the features of a layer to the loss.
            assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
            coeff = settings['features'][layer_name]
            x = layer_dict[layer_name].output
            # We avoid border artifacts by only involving non-border pixels in the loss.
            scaling = K.prod(K.cast(K.shape(x), 'float32'))
            if K.image_data_format() == 'channels_first':
                loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
            else:
                loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

        # Compute the gradients of the dream wrt the loss.
        grads = K.gradients(loss, dream)[0]
        # Normalize gradients.
        grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

        # Set up function to retrieve the value
        # of the loss and gradients given an input image.
        outputs = [loss, grads]
        fetch_loss_and_grads = K.function([dream], outputs)

    def eval_loss_and_grads(x):
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1]
        return loss_value, grad_values


    def resize_img(img, size):
        img = np.copy(img)
        if K.image_data_format() == 'channels_first':
            factors = (1, 1,
                       float(size[0]) / img.shape[2],
                       float(size[1]) / img.shape[3])
        else:
            factors = (1,
                       float(size[0]) / img.shape[1],
                       float(size[1]) / img.shape[2],
                       1)
        return scipy.ndimage.zoom(img, factors, order=1)


    def gradient_ascent(x, iterations, step, max_loss=None):
        for i in range(iterations):
            loss_value, grad_values = eval_loss_and_grads(x)
            if max_loss is not None and loss_value > max_loss:
                break
            print('..Loss value at', i, ':', loss_value)
            x += step * grad_values
        return x

    def dream():
        
        # Playing with these hyperparameters will also allow you to achieve new effects
        step = 0.01  # Gradient ascent step size
        num_octave = 3  # Number of scales at which to run gradient ascent
        octave_scale = 1.4  # Size ratio between scales
        iterations = 20  # Number of ascent steps per scale
        max_loss = 10.

        img = preprocess_image(self.base_image_path)
        if K.image_data_format() == 'channels_first':
            original_shape = img.shape[2:]
        else:
            original_shape = img.shape[1:3]
        successive_shapes = [original_shape]
        for i in range(1, num_octave):
            shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
            successive_shapes.append(shape)
        successive_shapes = successive_shapes[::-1]
        original_img = np.copy(img)
        shrunk_original_img = resize_img(img, successive_shapes[0])

        for shape in successive_shapes:
            print('Processing image shape', shape)
            img = resize_img(img, shape)
            img = gradient_ascent(img,
                                  iterations=iterations,
                                  step=step,
                                  max_loss=max_loss)
            upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
            same_size_original = resize_img(original_img, shape)
            lost_detail = same_size_original - upscaled_shrunk_original_img

            img += lost_detail
            shrunk_original_img = resize_img(original_img, shape)

        #Add url to image 


api.add_resource(DeepDreamAPI, '/v1/deepdream', endpoint='deepdream')
#api.add_resource(JobAPI, '/job/api/v1.0/jobs/<int:model_id>', endpoint='job')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')