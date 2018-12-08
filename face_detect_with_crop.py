#Import library required for Capture face.
# Should you wish to use this code for 
#education purpose in your assignment or dissertation
# please use the correct citation and give credit where required. 


import cv2
size = 4
i=1
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
face_cascade = cv2.CascadeClassifier('/home/hasank/Downloads/haarcascade_frontalface_default.xml')
#  Above line normalTest
#classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
#Above line test with different calulation
#classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
#classifier = cv2.CascadeClassifier('lbpcascade_frontalface.xml')


while True:
    (rval, img) = webcam.read()
    img=cv2.flip(img,1,0) #Flip to act as a mirror

    # Resize the image to speed up detection
    #mini = cv2.resize(im, (int(im.shape[1] / size),int( im.shape[0] / size))

    # detect MultiScale / faces 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around each face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        #cv2.rectangle(im, (x, y), (x + w, y + h),(0,255,0),thickness=4)
        #Save just the rectangle faces in SubRecFaces
        
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = img[ny:ny+nr, nx:nx+nr]
        lastimg = cv2.resize(faceimg, (500, 500))
        i += 1

        #cv2.imwrite("image%d.jpg" % i, lastimg)
        cv2.imwrite("i.jpg", lastimg)
    
    # Show the image
    cv2.imshow('BCU Research by Waheed Rafiq (c)',   img)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
