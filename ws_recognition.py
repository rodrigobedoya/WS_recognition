import face_recognition
from flask import Flask, jsonify, request, redirect
import numpy as np
import glob, os
import math

#for testing purposes, this chooses the max ammount of people analysed, total is about 5700
limit = 6000

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
	# Check if a valid image file was uploaded
	if request.method == 'POST':
		if 'file' not in request.files:
			return redirect(request.url)

		file = request.files['file']
		if file.filename == '':
			return redirect(request.url)

		data = request.form
		quantity = data['quantity']


		if file and allowed_file(file.filename):
			# The image file seems valid! Detect faces and return the result.
			return detect_faces_in_image(file,int(quantity))

	# If no valid image file was uploaded, show the file upload form:
	return '''
	<!doctype html>
	<title>Personas parecidas</title>
	<h1>Carga una foto y ve las personas que más se parecen!</h1>
	<form method="POST" enctype="multipart/form-data">
	  <input type="file" name="file">
	  <input type="submit" value="Cargar">
	  <br><br>
	  K: <input type="number" name="quantity">
	</form>
	'''

#create encoding of each image, save them in image folder
def createFaceEncodings():
	current_path = os.path.abspath(".")
	i = 0

	for file in os.listdir("static"):
		if i == limit:
			break;
		name = file.replace("_", " ")
		
		print("Analizing Person ",i+1, name)
		for image in os.listdir("static/"+file):
			#encode only jpg files
			if image.split(".")[1] == "jpg":
				imageFile = "static/"+file+"/"+image
				new_picture = face_recognition.load_image_file(imageFile)		
				new_face_encoding = face_recognition.face_encodings(new_picture)
				#If a face was found, save the encoding
				if len(new_face_encoding)>0:
					newFileName = image.split(".")[0]
					np.save(os.path.join('static/'+file, newFileName), new_face_encoding[0])
		i+=1

#manually calc euclidean distance
def distanciaEuclideana(v1,v2):
	result = 0
	for i in range(0,len(v1)):
		tmp = v1[i]-v2[i]
		tmp *= tmp
		result += tmp
	result = math.sqrt(result)
	return result

def detect_faces_in_image(file_stream,quantity):
	current_path = os.path.abspath(".")
	
	#encode user submitted image
	img = face_recognition.load_image_file(file_stream)
	unknown_face_encodings = face_recognition.face_encodings(img)
	
	if len(unknown_face_encodings) < 0:
		return "Couldn't find face"

	#uncomment lone below if first time running
	#createFaceEncodings()

	#distances calculated by library
	distances = []

	#distances calculated manually
	distances2 = []

	i = 0
	for file in os.listdir("static"):
			if i == limit:
				break;
			name = file.replace("_", " ")
			
			minDist = {}
			minDist['distance'] = -1
			
			minDist2 = {}
			minDist2['distance'] = -1
			
			print("Comparing to Person ",i+1, name)
			for image in os.listdir("static/"+file):
				if image.split(".")[1] == "jpg":
					imageFile = "static/"+file+"/"+image
					npFileName = current_path + "/static/"+file+"/"
					newFileName = image.split(".")[0]
					npFileName += newFileName + ".npy"

					#if there is a saved encoding of the image
					exists = os.path.isfile(npFileName)
					if exists:

						new_face_encoding = []
						new_face_encoding.append(np.load(npFileName))
						
						if len(new_face_encoding)>0:
							#add library distance
							distance = {}
							distance['name'] = name
							distance['image'] = imageFile
							output = face_recognition.face_distance(new_face_encoding, unknown_face_encodings[0])
							distance['distance'] = output[0]
							
							#add manually calculatedd distance
							distance2 = {}
							distance2['name'] = name
							distance2['image'] = imageFile
							eucl = distanciaEuclideana(new_face_encoding[0],unknown_face_encodings[0])
							distance2['distance'] = eucl
							

							#save only smallest library distance
							if minDist['distance'] == -1:
								minDist = distance 
							else:
								if distance['distance'] < minDist['distance']:
									minDist = distance 
							
							#save only smallest manually calculated distance
							if minDist2['distance'] == -1:
								minDist2 = distance2 
							else:
								if distance2['distance'] < minDist2['distance']:
									minDist2 = distance2 
			
			#if person has no available image encoding
			if minDist['distance'] != -1:
				distances.append(minDist)
			if minDist2['distance'] != -1:
				distances2.append(minDist2)
			i+=1

				
	#sort library distances
	sortedDistances = sorted(distances,key=lambda k: k['distance'])
	
	#sort manually calc'd distances
	sortedDistances2 = sorted(distances2,key=lambda k: k['distance'])
	
	
	#GENERATING HTML template
	htmlCodeStart = """
	<!DOCTYPE html>
	<html>
	<head>
	<title>Tarea 2 Base de Datos II</title>
	</head>
	<body>
	"""

	htmlNameBegin = '''
	<h2>'''

	htmlNameEnd = ''': </h2>
	'''
	htmlImageBegin = '''
	<img src="/'''
	
	htmlImageEnd = '''" >
	<br><br>
	'''
	htmlCodeEnd = """
	</body>
	</html>
	"""

	#creating start of html file
	htmlCode = htmlCodeStart
	htmlCode += """<h1>Top """
	htmlCode += str(quantity)
	htmlCode += """ Personas que más se parecen</h1>"""
	
	print()
	print("RESULTS FACE_RECOGNITION FUNCTION")
	for i in range(0,quantity):
		#add name of person and photo to html
		htmlCode += htmlNameBegin
		htmlCode += str(i+1) + ". "
		htmlCode += sortedDistances[i]['name']
		htmlCode += htmlNameEnd
		htmlCode += htmlImageBegin
		htmlCode += sortedDistances[i]['image']
		htmlCode += htmlImageEnd
		print(sortedDistances[i])
	
	#finish html template
	htmlCode += htmlCodeEnd;

	print()
	print("RESULTS EUCLIDEAN DISTANCE")
	for i in range(0,quantity):
		print(sortedDistances2[i])

	return htmlCode

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5001, debug=True)
