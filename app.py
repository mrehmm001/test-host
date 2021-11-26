from flask import Flask, render_template,request, send_file
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import io
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from skimage.io import imshow,imsave
import base64

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("colourise.html")

@app.route("/colourise",methods=["POST","GET"])
def colourise():
    if request.method=="POST":
        image=[]
        img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
        original_size = img.shape
        print(img.shape)
        img = cv2.resize(img,(256,256))
        image.append(img)
        image = np.array(image,dtype=float)
        image = rgb2lab(1.0/255*image)[:,:,:,0]
        image = image.reshape(image.shape+(1,))
        model = tf.keras.models.load_model('./static/colorize_autoencoder.model4')
        prediction = model.predict(image)
        prediction=prediction*128
        result = np.zeros((256,256,3))
        result[:,:,0]=image[0][:,:,0]
        result[:,:,1:]=prediction[0][:,:,0:]
        result = lab2rgb(result)
        result = cv2.resize(result,(original_size[1],original_size[0]))

        result=result*255
        print(result.shape)
        imsave("test.jpg",result)
        result = Image.fromarray(result.astype("uint8"))
        rawBytes = io.BytesIO()
        result.save(rawBytes, "JPEG")
        result_base64 = base64.b64encode(rawBytes.getvalue())

        im = Image.open(request.files["image"])
        data = io.BytesIO()
        im.save(data,"JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())

        
        return render_template("index.html",result=result_base64.decode('utf-8'),before=encoded_img_data.decode('utf-8'))
    return "."
    
if __name__ == "__main__":
    app.run(debug=True)