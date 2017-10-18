from flask import Flask, request
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

todos = {}

class JobAPI(Resource):
    def get(self, todo_id):
        return {todo_id: todos[todo_id]}

    def put(self, todo_id):
        todos[todo_id] = request.form['data']
        return {todo_id: todos[todo_id]}

class JobListAPI(Resource):
    def get(self):
        return todos 

api.add_resource(Tasks, '/<string:todo_id>')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
