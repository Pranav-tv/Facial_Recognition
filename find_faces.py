'''################################ Face Detection ################################'''


import cv2

### Load OpenCV Classifier

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

### Load Image

img = cv2.imread('Training Data/6.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

### Face Detection

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print('Faces found: ', len(faces))

cv2.startWindowThread()
cv2.imshow('Original Image',img)  ### Display Image
cv2.waitKey(3000)
cv2.destroyAllWindows()


for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)   ### Draw rectangles around face
    
    roi_gray = gray.copy()
    roi_gray = roi_gray[y:y+h, x:x+w] 
    roi_color = img.copy()          ### Cropping the rectangele area
    roi_color = roi_color[y:y+h, x:x+w]   ### Cropping the rectangele area
    
### Eye Detection
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        eyes = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

# Display detected faces
cv2.startWindowThread()
cv2.imshow('Detected Faces',img)
cv2.waitKey(5000)
cv2.destroyAllWindows() 


#Display detected eyes of faces
cv2.startWindowThread()
cv2.imshow('Detected Eyes',eyes)
cv2.waitKey(5000) 
cv2.destroyAllWindows()












