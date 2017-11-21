from flask import Flask, request, url_for, abort
from flask_restful import Resource, Api, reqparse
import keras
from keras.models import Sequental

app = Flask(__name__)
api = Api(app)

jobs = {}
            
class SequentialModelAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('title', type = str, location = 'json')
        self.reqparse.add_argument('description', type = str, loctiton = 'json')
        self.reqparse.add_argument('done', type = bool, location = 'json')
        super(JobAPI, self).__init__()

    def get(self, job_id):
        job = [job for job in jobs if job['job_id'] == job_id]
        if len(job) == 0:
        	abort(404, message="Job {} does not exist".format(job_id))
        return { 'job': id_to_uri(job) }

    def patch(self, job_id):

    def put(self, job_id):
        args = self.reqparse.parse_args()
        job = jobs[job_id]
        for k, v in args.iteritems():
            if v != None:
                job[k] = v
        return {job_id: jobs[job_id]}

    def delete(self, job_id):

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
#api.add_resource(JobAPI, '/job/api/v1.0/jobs/<int:job_id>', endpoint='job')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')