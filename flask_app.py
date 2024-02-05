from flask import Flask, request, jsonify
from flasgger import Swagger
import os
from flask_cors import CORS
from preprocess import preprocess
from PIL import Image

# from eigenfaces import eigenFaces  # Assuming you have an eigenFaces module
from fisherface import fisherFaces  # Assuming you have a fisherFaces module

app_root = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)


def recognize_face(image_data):
    try:
        # Preprocess the image data using the preprocess function
        preprocessed_data = preprocess(image_data)

        # Call eigenFaces predict function
        # eigenfaces_status, eigenfaces_name = eigenFaces.predict(preprocessed_data)

        # Call fisherFaces predict function
        fisherfaces_name = fisherFaces.predict(preprocessed_data)

        # Return the combined results
        result = {
            "eigenfaces": {"status": "Known Face", "predicted_name": "Fasika"},
            "fisherfaces": {"predicted_name": fisherfaces_name},
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/")
def hello_world():
    return "Hello from Fasika!"


@app.route("/recognizeface", methods=["POST"])
def recognize_face_route():
    try:
        # Assuming the image data is sent as base64 encoded in the request
        image_data = request.json.get("image_data")

        # Call the main face recognition function
        recognition_results = recognize_face(image_data)

        return recognition_results
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    with Image.open("ep.jpg") as img:
        print(recognize_face(img))
