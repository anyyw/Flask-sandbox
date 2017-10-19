from flask import Flask, request
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

jobs = {}

class JobAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('title', type = str, location = 'json')
        self.reqparse.add_argument('description', type = str, loctiton = 'json')
        self.reqparse.add_argument('done', type = bool, location = 'json')
        super.(JobAPI, self).__init__()

    def get(self, job_id):
        return {job_id: jobs[job_id]}

    def put(self, job_id):
        jobs[job_id] = request.form['data']
        return {job_id: jobs[job_id]}

class JobListAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('title', type = str, required = True,
            help = 'No Job title provided', location = 'json')
        self.reqparse.add_argument('description', type = str, default = "", location = 'json')
        super.(JobListAPI, self).__init__()

    def get(self):
        return jobs 

api.add_resource(JobListAPI, '/job/api/v1.0/jobs', endpoint='jobs')
api.add_resource(JobAPI, '/job/api/v1.0/jobs/<int:job_id>', endpoint='job')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
