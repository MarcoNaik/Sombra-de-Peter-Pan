import cv2
from os import listdir
from os.path import isfile, join
import os
import random
import numpy as np
import dlib

source_videos_path = 'driving_video'
driving_videos = [f for f in listdir(source_videos_path) if isfile(join(source_videos_path, f))]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
border_amount = 0.15

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

def runFOM(video):
    fom_command = "python demo.py  --config config/vox-adv-256.yaml --driving_video %s --source_image img.png --checkpoint fom_checkpoints/vox-adv-cpk.pth.tar --relative --adapt_scale"%video
    os.system(fom_command)

def detectFaces(frame):
    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detector(gray)
    if len(face)<1:
        return None
    return (face[0], gray)

def get_driving_video():
    return random.choice(driving_videos)

def is_any_negative(list):
    return True in (ele < 0 for ele in list)

def show_previous_cropped_frame(frame, state):
    cropped_frame = crop(frame, state.cropped_face_cords)
    cv2.imshow("obra", cropped_frame)
    state.statemachine.current_video.write(cropped_frame)

def new_cropped_face_cords_func(frame, state):
    faces = detectFaces(frame)


    if faces is None:
        show_previous_cropped_frame(frame, state)
        return None

    face, gray = faces
    new_cropped_face_cords = broaderFaceCords(face)

    if is_any_negative(new_cropped_face_cords):
        print("negative on cropped face cords")
        show_previous_cropped_frame(frame, state)
        return None
    x1,y1,x2,y2 = new_cropped_face_cords

    #cv2.imshow("frame_debug", cv2.rectangle(np.copy(frame), (x1, y1), (x2, y2), (0, 255, 0), 3))
    state.cropped_face_cords = new_cropped_face_cords
    return (new_cropped_face_cords, face, gray)
