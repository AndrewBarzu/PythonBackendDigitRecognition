from flask import Flask
from flask import request
from flask_restful import Resource, Api
from flask_jsonpify import jsonify
import numpy as np
import scipy.io as sio
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

Theta1 = sio.loadmat('Theta1.mat')
Theta2 = sio.loadmat('Theta2.mat')
Theta1 = np.array(Theta1['Theta1'])
Theta2 = np.array(Theta2['Theta2'])

class NeuralNetwork3Layers(Resource):
    def __init__(self):
        self._Theta1 = Theta1
        self._Theta2 = Theta2

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _predict(self, parameters):
        if parameters is None:
            return None
        parameters = np.array(parameters)
        X = np.concatenate((np.ones(1), parameters))[np.newaxis]
        a2 = self._sigmoid(np.dot(X, self._Theta1.transpose())).transpose()
        a2 = np.concatenate((np.ones(a2.shape[1])[np.newaxis], a2)).transpose()
        a3 = self._sigmoid(np.dot(a2, self._Theta2.transpose()))
        return (np.argmax(a3) + 1) % 10

    def get(self):
        image = request.args.get('image')
        image = list(map(float, image.split(',')))
        image = np.array(image)
        result = self._predict(image)
        print(result)
        return jsonify(str(result))

api.add_resource(NeuralNetwork3Layers, '/prediction')


if __name__ == '__main__':
    app.run()
