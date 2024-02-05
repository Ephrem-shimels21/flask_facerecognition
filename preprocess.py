import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def detect_face(image_array):
    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.05,
        minNeighbors=10,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) > 0:
        # Get the coordinates of the first detected face
        x, y, w, h = faces[0]

        # Crop the face from the original image
        face = image_array[y : y + h, x : x + w]

        return face
    else:
        print("returning none in detection")
        return image_array


#     return output_image
def preprocess(img):
    img_array = np.array(img)
    face = detect_face(img_array)
    output_face = Image.fromarray(face)

    output_image = output_face.resize((150, 150))
    output_image = output_image.convert("L")
    output_array = np.array(output_image)

    return output_array


preprocess(Image.open("ep.jpg"))
