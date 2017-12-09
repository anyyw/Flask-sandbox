#!flast/bin/python

from flask import Flask, request, url_for, abort
from flask_restful import Resource, Api, reqparse
import keras
from keras.models import Sequental

app = Flask(__name__)
api = Api(app)

models = {}
            
class SequentialModelAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('title', type = str, location = 'json')
        self.reqparse.add_argument('description', type = str, loctiton = 'json')
        self.reqparse.add_argument('done', type = bool, location = 'json')
        super()

    def get(self, model_id):
        model = [model for model in models if model['model_id'] == model_id]
        if len(model) == 0:
        	abort(404, message="Sequential Model {} does not exist".format(model_id))
        return { 'model': id_to_uri(model) }

    def patch(self, model_id):

    def post(self, model_id):
        args = self.reqparse.parse_args()
        model = models[model_id]
        for k, v in args.iteritems():
            if v != None:
                model[k] = v


        return {model_id: model[model_id]}

    def delete(self, model_id):

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