from flask import Flask,jsonify,request
from flasgger import Swagger
# from sklearn.externals import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
Swagger(app)
CORS(app)

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    iris_dataset.data, iris_dataset.target, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train,y_train)

@app.route("/", methods=['POST'])
def predict():
    """
    Ini Adalah Endpoint Untuk Memprediksi IRIS
    ---
    tags:
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: Petal
          required:
            - petalLength
            - petalWidth
            - sepalLength
            - sepalWidth
          properties:
            petalLength:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
            petalWidth:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
            sepalLength:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
            sepalWidth:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()

    petalLength = new_task['petalLength']
    petalWidth = new_task['petalWidth']
    sepalLength = new_task['sepalLength']
    sepalWidth = new_task['sepalWidth']

    X_New = np.array([[petalLength,petalWidth,sepalLength,sepalWidth]])

    # clf = joblib.load('knnClasifier.pkl')

    resultPredict = clf.predict(X_New)

    return jsonify({'message': format(iris_dataset.target_names[resultPredict])})