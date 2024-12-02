from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras_facenet import FaceNet
import pickle
import os

app = Flask(__name__)

try:
    with open("data.pkl", "rb") as myfile:
        database = pickle.load(myfile)
except FileNotFoundError:
    raise RuntimeError("Error: 'data.pkl' not found! Ensure the file is in the correct location.")

MyFaceNet = FaceNet()

HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if HaarCascade.empty():
    raise RuntimeError("Error loading Haar Cascade for face detection!")

def recognize_face(img):
    gbr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gbr_array = np.asarray(gbr)

    wajah = HaarCascade.detectMultiScale(gbr_array, 1.1, 4)
    
    if len(wajah) == 0:
        return "No Face Detected"
    
    x1, y1, w, h = wajah[0]
    face = gbr_array[y1:y1+h, x1:x1+w]
    face = cv2.resize(face, (160, 160)) 
    face = np.expand_dims(face, axis=0)
    
    signature = MyFaceNet.embeddings(face)
    
    min_dist = 100
    identity = "Unknown"
    
    for key, value in database.items():
        dist = np.linalg.norm(value - signature / np.linalg.norm(signature))
        if dist < min_dist:
            min_dist = dist
            identity = key

    return identity

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Service is live!'}), 200

@app.route("/recognize", methods=["POST"])
def recognize():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    result = recognize_face(img)
    
    return jsonify({"result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)