import cv2
import os
import numpy as np

#################################################################################

#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Messi", "Ronaldo"]

#################################################################################

#function to detect face using OpenCV

def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    return gray,faces

#################################################################################

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list of faces and another list of #labels for each face
    
def prepare_training_data(data_folder_path):
 
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
     
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
     
        #build path of directory containing images for current subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
         
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
     
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
         
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
         
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name
        
            #read image
            image = cv2.imread(image_path)
         
            #display an image window to show the image 
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
         
            #detect face
            gray_img, faces_array = detect_face(image)
            
            for i in range(len(faces_array)):
                (x, y, w, h) = faces_array[i]
                face = gray_img[y:y+w, x:x+h]
         
                #------STEP-4--------
                #for the purpose of this tutorial
                #we will ignore faces that are not detected
                if face is not None:
                    #add face to list of faces
                    faces.append(face)
                    #add label for this face
                    labels.append(label)
             
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
             
    return faces, labels

#################################################################################

data_folder_path = '/home/pranav/Desktop/Data Science/DEEP LEARNING/Basic Facial Recog System/Face Recognition/Training Data'

print("Preparing data...")
faces, labels = prepare_training_data(data_folder_path)
print("Data prepared")
 
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# Training the Face Recognition Model

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#To use EigenFaceRecognizer replace above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#To use FisherFaceRecognizer replace above line with 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

#################################################################################

#function to draw rectangle on image 

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
#function to draw text on give image starting from passed (x, y) coordinates
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
#################################################################################

#Prediction Function
    
#this function recognizes the person in image passed and draws a rectangle around detected face with name of the subject
    
def predict(test_img):

    img = test_img.copy()
    gray_img, faces_array = detect_face(img)
    
    for i in range(len(faces_array)):
        (x, y, w, h) = faces_array[i]
        face = gray_img[y:y+w, x:x+h]
        rect = faces_array[i]
 
        #predict the image using our face recognizer 
        label= face_recognizer.predict(face)
        
        #get name of respective label returned by face recognizer
        label_text = subjects[label[0]]
         
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
     
    return img    

#################################################################################
    
print ("Predicting images...")
 
#load test images

test_img1 = cv2.imread('Training Data/1.jpeg')
test_img2 = cv2.imread('Training Data/2.jpeg')
test_img3 = cv2.imread('Training Data/3.jpeg')
test_img4 = cv2.imread('Training Data/4.jpeg')
test_img5 = cv2.imread('Training Data/5.jpeg')
 
#perform a prediction

predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)
predicted_img5 = predict(test_img5)
print("Prediction complete")
 
#display both images
cv2.imshow('img',predicted_img1)
cv2.waitKey(1000)
cv2.imshow('img',predicted_img2)
cv2.waitKey(1000)
cv2.imshow('img',predicted_img3)
cv2.waitKey(1000)
cv2.imshow('img',predicted_img4)
cv2.waitKey(1000)
cv2.imshow('img',predicted_img5)
cv2.waitKey(1000)
cv2.destroyAllWindows()