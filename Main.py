import numpy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import time
import cv2
from kivy.uix.image import Image

Builder.load_string('''
<FaceDetection>:
    orientation: 'vertical'
    BoxLayout:
        id: camera
        resolution: (640, 480)
    ToggleButton:
        text: 'Play'
        on_press: root.face()
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
''')


class FaceDetection(BoxLayout):
    def face(self):
        camera = self.ids['camera']
        #img = cv2.flip(img, 1, 0)
        i = 1
        webcam = cv2.VideoCapture(0)  # Use camera 0

        # We load the xml file
        face_cascade = cv2.CascadeClassifier('/home/hasank/Downloads/haarcascade_frontalface_default.xml')

        while True:
            (rval, img) = webcam.read()
            img = cv2.flip(img, 1, 0)  # Flip to act as a mirror

            # detect MultiScale / faces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Draw rectangles around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                r = max(w, h) / 2
                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int(r * 2)

                faceimg = img[ny:ny + nr, nx:nx + nr]
                lastimg = cv2.resize(faceimg, (500, 500))
                i += 1
                cv2.imwrite("i.jpg", lastimg)

            # Show the image
            cv2.imshow(camera, img)
            key = cv2.waitKey(10)

            # if Esc key is press then break out of the loop
            if key == 27:  # The Esc key
                break


class CameraClick(BoxLayout):
    def capture(self):
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")


class TestCamera(App):

    def build(self):
        return FaceDetection()


TestCamera().run()
