import os
import cv2
import numpy as np
import dlib

#cam = cv2.VideoCapture(0)
#cv2.namedWindow("test")
border_amount = 0.15
#width = 1280
#height = 720
#cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

fom_command = "python demo.py  --config config/vox-adv-256.yaml --driving_video driving_video/ --source_image img.png --checkpoint fom_checkpoints/vox-adv-cpk.pth.tar --relative --adapt_scale"
faces = 'img.png'
NNcap = cv2.VideoCapture("result.mp4")

def broaderFaceCords(d):
    x1, y1, x2, y2 = (d.left(), d.top(), d.right(), d.bottom())
    w = abs(x2-x1)
    h = abs(y2-y1)

    x1 -= w*border_amount
    y1 -= h*border_amount
    x2 += w*border_amount
    y2 += h*border_amount

    return list(map(int, (x1, y1, x2, y2)))

def crop(frame, cords):
    x1,y1,x2,y2 = cords
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (256, 256))

def writePhoto(cropped_face):
    #save as NN input
    if not cv2.imwrite("img.png", cropped_face):
        raise Exception('couldnt  write img')

def runFOM():
    os.system(fom_command)

def detectFaces(frame):
    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detector(gray)
    if len(face)<1:
        return None
    return face[0]
"""
delay = 0
photoTaken=False
while True:
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        cv2.destroyAllWindows()
        break
    #Get frame
    ret, frame = cam.read()
    #Check
    if not ret:
        print("failed to grab frame")
        break
    #Render on canvas
    cv2.imshow("test", frame)
    #Search for faces
    faceData = detectFaces(frame)
    #print(faceData)
    if faceData is not None:
        #Take photo
        delay+= 1
        print(delay)
        if delay < 10 :
            continue
        if not photoTaken:
            #Save input
            takephoto(faceData, frame)
            #Run NN
            print("FOM")
            #runFOM()
            print("FOMdone")
            #render mp4 to frames
            photoTaken = True
        #posicionar video encima de carita
        else:
            #show
            ret_val, NNframe = NNcap.read()
            #blend = cv2.addWeighted(frame, 0.5, NNframe, 0.5, 0.0)
            #cv2.imshow("test", NNframe)
            x1,y1,x2,y2 = broaderFaceCords(faceData, border_amount)
            frame[y1:y2, x1:x2] =  cv2.resize(NNframe, (x2-x1, x2-x1))
            cv2.imshow("test", frame)
            #get pos cara
    #ret_val, NNframe = NNcap.read()
    #cv2.imshow("test", NNframe)

cam.release()
cv2.destroyAllWindows()
"""
