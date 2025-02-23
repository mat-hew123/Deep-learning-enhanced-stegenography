import numpy as np 
from PIL import Image, ImageOps 
from flask import Flask, render_template, request 

### Imports ###
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.misc
from tqdm import *

import io
from PIL import Image
import base64
import uuid
from PIL import ImageEnhance, ImageFilter


beta = 1.0

# Loss for reveal network
def rev_loss(s_true, s_pred):
    # Loss for reveal network is: beta * |S-S'|
    return beta * K.sum(K.square(s_true - s_pred))

# Loss for the full model, used for preparation and hidding networks
def full_loss(y_true, y_pred):
    # Loss for the full model is: |C-C'| + beta * |S-S'|
    s_true, c_true = y_true[..., 0:3], y_true[..., 3:6]
    s_pred, c_pred = y_pred[..., 0:3], y_pred[..., 3:6]

    s_loss = rev_loss(s_true, s_pred)
    c_loss = K.sum(K.square(c_true - c_pred))
    return s_loss + c_loss

from tensorflow.keras.models import load_model
autoencoder_model = load_model("model.h5", custom_objects={'full_loss': full_loss})
autoencoder_model.load_weights("model_weights_best.hdf5")


def steganography(secret, cover):
    
    secret_img = Image.open(io.BytesIO(secret.read()))
    cover_img = Image.open(io.BytesIO(cover.read()))
    
    secret_image = secret_img.resize((64, 64))
    cover_image = cover_img.resize((64, 64))
    
    # Convert images to arrays and normalize
    secret_image = image.img_to_array(secret_image) / 255.0
    cover_image = image.img_to_array(cover_image) / 255.0

    # Reshape images to match model input
    secret_image = np.expand_dims(secret_image, axis=0)
    cover_image = np.expand_dims(cover_image, axis=0)

    # Encode secret image within cover image using the autoencoder model
    encoded_image = autoencoder_model.predict([secret_image, cover_image])[..., 3:6]  # Extracting the encoded image

    # Decode the encoded image to reveal the hidden secret
    decoded_image = autoencoder_model.predict([secret_image, cover_image])[..., 0:3]  # Decoding the hidden image

        
    #encoded_image_pil = Image.fromarray((encoded_image[0] * 255).astype(np.uint8))
    #encoded_image_enhanced = ImageEnhance.Contrast(encoded_image_pil).enhance(1.5)  # Example enhancement (increase contrast)
    
    # Post-process the decoded image to enhance its quality
    #decoded_image_pil = Image.fromarray((decoded_image[0] * 255).astype(np.uint8))
    #decoded_image_enhanced = ImageEnhance.Contrast(decoded_image_pil).enhance(1.5)  # Example enhancement (increase contrast)
    # Convert the enhanced images back to numpy arrays
    #encoded_image_enhanced_np = np.array(encoded_image) / 255.0
    #decoded_image_enhanced_np = np.array(decoded_image) / 255.0
    return encoded_image, decoded_image



app = Flask(__name__, static_url_path='/static') 

app.config['UPLOAD_FOLDER'] = 'static'



@app.route('/', methods=['GET', 'POST']) 
def index(): 
    if request.method == 'POST': 
        secret = request.files['secret']
        cover = request.files['cover']
        function = request.form['function']
        print(function)
        stego_img, decoded_img= steganography(secret, cover)
        
        # Generate unique filenames for the stego image and decoded image
        stego_filename = str(uuid.uuid4()) + '.png'
        decoded_filename = str(uuid.uuid4()) + '.png'
        
        # Save the stego image and decoded image to the static folder
        stego_img_path = os.path.join(app.config['UPLOAD_FOLDER'], stego_filename)
        decoded_img_path = os.path.join(app.config['UPLOAD_FOLDER'], decoded_filename)
        
        stego_img_pil = Image.fromarray((stego_img[0] * 255).astype(np.uint8))
        decoded_img_pil = Image.fromarray((decoded_img[0] * 255).astype(np.uint8))
        
        stego_img_pil.save(stego_img_path)
        decoded_img_pil.save(decoded_img_path)
        
        if(function=="steganography"):
            return_filename=stego_filename
        else:
            return_filename=decoded_filename
        return render_template('indexT.html', return_image=return_filename)
    return render_template('indexT.html') 


if __name__ == '__main__': 
	app.run(debug=True) 
