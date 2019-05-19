from flask import Flask, redirect, url_for, request,render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os
app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/", methods = ['GET'])
def index():
    return render_template('index.html')

@app.route("/details", methods = ['GET'])
def index2():
    return render_template('index2.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		print("hi this is a post")
		f = request.files['file']
		f.save("test.jpg")
		os.system("python3 inference.py")
		f = open("output/details.txt","r")
		count = 0
		for x in f.readlines():
			print(x)
			count+=1
		if count>=1:
			return str(count)+" dogs are found in the provided image"
		return "Please assure Image contains a dog"
	return None
# app.run(host='0.0.0.0', port=50000)

if __name__=='__main__':
	http_server = WSGIServer(('',50030),app)
	http_server.serve_forever()