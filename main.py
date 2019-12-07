from flask import Flask, request, render_template, send_from_directory
from flask_restful import Resource, Api
import gunicorn
import controller as ctrl
import urllib.request

app = Flask(__name__)
app.static_folder = 'static'
AppApi = Api(app)
class twpy:
    def __init__(self):
        print("init called")
        try:

            self.api = ctrl.connectTwitter()
            print("Connected with Twitter !")
        except:
            print ('!!! Error Connecting to Twitter !!!')

    def getapi(self):
        print(self.api)
        return self.api


class Hello(Resource):
    def get(self):
        text = "Demonitization"
        try:
            driver = ctrl.connectNeo4j()
            print("Connected with Neo4j !")
            session=driver.session()
            session.write_transaction(ctrl.delAlledges)
            print ('All Edges Deleted')
        except:
            print ('!!! Error Connecting to Neo4j !!!')
        api =twpy().getapi()
        res = ctrl.extract(session,text,api)
        topics = res[0]
        for names in topics.keys():
            print (names)
        return topics
     

# class MyApi(Resource):
#     def post(self):

#         some_json = request.args
#         print(some_json)

AppApi.add_resource(Hello, '/')
# AppApi.add_resource(MyApi, '/extract_datad')

if __name__ == "__main__":
    app.run(debug = True)
    