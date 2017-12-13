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

app = Flask(__name__)
api = Api(app)

models = {}
jobs = {}

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

'''
class JobListAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('title', type = str, required = True,
            help = 'No Job title provided', location = 'json')
        self.reqparse.add_argument('description', type = str, default = "", location = 'json')
        super(JobListAPI, self).__init__()

    def get(self):
        return jobs 

'''

api.add_resource(SequentialModelAPI, '/v1/sequentialmodel', endpoint='sequential')
#api.add_resource(JobAPI, '/job/api/v1.0/jobs/<int:model_id>', endpoint='job')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')