from flask import Flask, request, url_for
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

jobs = {}

def id_to_uri(job):
    new_job = {}
    for field in job:
        if field == 'id':
            new_job['uri'] = url_for('get_job', job_id=job['id'], _external=True)
        else:
            new_job[field] = job[field]
    return new_job

def validate_jobid_exists(job_id):
        if job_id not in jobs:
            abort(404, message="Job {} does not exist".format(job_id))

class JobAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('title', type = str, location = 'json')
        self.reqparse.add_argument('description', type = str, loctiton = 'json')
        self.reqparse.add_argument('done', type = bool, location = 'json')
        super.(JobAPI, self).__init__()

    def get(self, job_id):
        validate_jobid_exists(job_id)
        job = jobs[job_id]
        return { 'job': id_to_uri(job) }

    def put(self, job_id):
        validate_jobid_exists(job_id)
        args = self.reqparse.parse_args()
        job = jobs[job_id]
        for k, v in args.iteritems():
            if v != None:
                job[k] = v
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
