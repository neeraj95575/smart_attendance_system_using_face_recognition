from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from numpy import asarray
from PIL import Image
import glob
import time 
import cv2 
import os


########################################## CLICK THE PHOTO using camera ############################

timer = int(5) # timer

cap = cv2.VideoCapture(0) 
while True: 
	ret, img = cap.read() 
	cv2.imshow('a', img)  
	prev = time.time() 

	while timer >= 0: 
			ret, img = cap.read() 
			font = cv2.FONT_HERSHEY_SIMPLEX 
			cv2.putText(img, str(timer), (200, 250), font, 7, (0, 255, 255), 4, cv2.LINE_AA) 
			cv2.imshow('a', img) 
			cv2.waitKey(125) 
			cur = time.time()
			
			if cur-prev >= 1: 
				prev = cur 
				timer = timer-1

	else: 
			ret, img = cap.read() 
			cv2.imshow('a', img) 
			cv2.waitKey(2000) 
			cv2.imwrite('clicked_photo/student.jpg', img)
			cap.release()
			break

########################################################################################


count = 0		    
def extract_face_from_image(image_path, required_size=(300, 300)):  #required_size=(224, 224))
  # load image and detect faces
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images

def get_model_scores(faces):
    samples = asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    model = VGGFace(model='resnet50',include_top=False,input_shape=(300, 300, 3),pooling='avg')

    # perform prediction
    return model.predict(samples)

new_faces = extract_face_from_image('clicked_photo/student.jpg')
old_faces = extract_face_from_image('images/photo.jpg')

model_scores_new = get_model_scores(new_faces)
model_scores_old = get_model_scores(old_faces)

for idx, face1_score in enumerate(model_scores_new):
  for idy, face2_score in enumerate(model_scores_old):
    score = cosine(face1_score, face2_score)
    if score <= 0.4:
      # Printing the IDs of faces and score
      print(idx, idy, score)
      plt.imshow(new_faces[idx])
      print("\n")
      print("saving images of students present in the class...................")
      print("\n")
      plt.savefig("present_students/"+str(count)+".jpg", dpi=500, bbox_inches='tight',pad_inches=0)      
      count +=1

print("all images are saved") 
