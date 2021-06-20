from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)

dic = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
dic2 = {0:'Female', 1:'Male'}

model = load_model('Model.h5')
model2 = load_model('Gender_Detection.h5')
orig_path = './static/image.png'

model.make_predict_function()
model2.make_predict_function()

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

def crop_img(img_path):
	cropped_image = get_cropped_image_if_2_eyes(img_path)
	cv2.imwrite('./static/image.png', cropped_image)

def predict_label1(img_path):
    i = image.load_img(img_path, target_size=(48, 48))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 48, 48, 3)
    p = model.predict_classes(i)
    return dic[p[0]]

def predict_label2(img_path):
    i = image.load_img(img_path, target_size=(48, 48))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 48, 48, 3)
    p = model2.predict_classes(i)
    return dic2[p[0]]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    try:
        if request.method == 'POST':
            img = request.files['my_image']
            img_path = "static/" + img.filename
            img.save(img_path)
            crop_img(img_path)
            p = predict_label1(orig_path)
            sex1 = predict_label2(orig_path)
            return render_template("index.html", prediction=p, sex= sex1, img_path2 = orig_path, img_path=img_path)
    except:
        return render_template("index.html", alert="Upload Image to Predict")


if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True)
