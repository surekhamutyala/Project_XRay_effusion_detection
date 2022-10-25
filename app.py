from tkinter import Image
from flask import Flask, render_template, request
import os
import pickle
from PIL import Image
import numpy as np
from skimage.transform import rescale

app = Flask(__name__)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT_PATH, "static", "images") # path where Uploaded images will be stored
MODEL_FOLDER = os.path.join(ROOT_PATH, "static", "XRay_effusion_detection.pickle") # path where model.pikle file stored

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER # It is used to mention the upload folder
app.config["MODEL_FOLDER"] = MODEL_FOLDER

def make_predictions():
    XRay_effusion_detection_model = pickle.load(open(MODEL_FOLDER, 'rb'))
    image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], "image_to_test.png"))
    image = preprocess_img(image)

    preds = XRay_effusion_detection_model.predict(image)
    return preds

def preprocess_img(img):
    img = (img - img.min())/(img.max() - img.min())
    img = rescale(img, 0.25, multichannel=True, mode='constant')
    return img


@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        # once we upload the file and send it to server
        image  = request.files["image"] # At the server side, the file is fetched using the request.files['file'] object and saved to the location on the server.
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], "image_to_test.png")) # name of the file uploaded will be 'image_to_test'.png
        preds = make_predictions()
        return render_template("display.html", preds = preds)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)