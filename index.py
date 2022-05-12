from keras.preprocessing.image import img_to_array 
from skimage.transform import resize
import numpy as np
import tensorflow as tf
from flask import Flask, render_template , request , jsonify
from skimage.io import imsave 
from keras.preprocessing.image import  img_to_array , load_img
from PIL import Image
import io , sys
import numpy as np 
import cv2
import base64
from skimage.color import rgb2lab , lab2rgb


app = Flask(__name__)

############################################## THE REAL DEAL ###############################################
@app.route('/colorize' , methods=['POST'])
def mask_image():
	model = tf.keras.models.load_model('colorize_v6.1',
                                		custom_objects = None,
                                		compile = True)

	# print(request.files , file=sys.stderr)
	img = request.files['image'].read() ## byte file
	npimg = np.fromstring(img, np.uint8)
	img = cv2.imdecode(npimg , cv2.IMREAD_COLOR)
	######### Do preprocessing here ################
	img_color =[]   
	img = img_to_array(img)
	img = resize(img , (256 , 256))
	img_color.append(img)

	img_color = np.array(img_color , dtype=float)
	img_color = rgb2lab(1.0 / 255 * img_color)[:,:,:,0]
	img_color = img_color.reshape(img_color.shape+(1,))

	output = model.predict(img_color)
	output = output * 128
	res = np.zeros((256 , 256, 3))
	res[:,:,0] = img_color[0][:,:,0]
	res[:,:,1:] = output[0]
	img = lab2rgb(res)
	imsave('output.jpeg' , img)
	img = img_to_array(load_img('output.jpeg'))

	################################################
	img = Image.fromarray(img.astype("uint8"))
	rawBytes = io.BytesIO()
	img.save(rawBytes, "JPEG")
	rawBytes.seek(0)
	img_base64 = base64.b64encode(rawBytes.read())
	return jsonify({'status':str(img_base64)})

##################################################### THE REAL DEAL HAPPENS ABOVE ######################################

# @app.route('/test' , methods=['GET','POST'])
# def test():
# 	print("log: got at test" , file=sys.stderr)
# 	return jsonify({'status':'succces'})

# @app.route('/home')
# def home():
# 	return render_template('index.jinja2')


	
@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run(debug = True)