from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras_facenet import FaceNet
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained FaceNet model and the database
with open("data.pkl", "rb") as myfile:
    database = pickle.load(myfile)
MyFaceNet = FaceNet()

# Haar Cascade for face detection
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

# Function to process the uploaded image and detect faces
def recognize_face(img):
    # Convert image to RGB
    gbr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gbr_array = np.asarray(gbr)

    # Detect faces using Haar Cascade
    wajah = HaarCascade.detectMultiScale(gbr_array, 1.1, 4)
    
    if len(wajah) == 0:
        return "No Face Detected"
    
    x1, y1, w, h = wajah[0]
    face = gbr_array[y1:y1+h, x1:x1+w]
    face = cv2.resize(face, (160, 160))  # Resize face to match model input
    face = np.expand_dims(face, axis=0)
    
    # Get the face embedding
    signature = MyFaceNet.embeddings(face)
    
    # Compare face embedding with database
    min_dist = 100
    identity = "Unknown"
    
    for key, value in database.items():
        dist = np.linalg.norm(value - signature)
        if dist < min_dist:
            min_dist = dist
            identity = key

    return identity

# Endpoint to recognize faces from uploaded image
@app.route("/recognize", methods=["POST"])
def recognize():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read the image file into OpenCV
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Process the image
    result = recognize_face(img)
    
    # Return the result as JSON
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
