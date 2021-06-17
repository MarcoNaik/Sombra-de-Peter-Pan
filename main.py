import cv2
from states import StateMachine
import time
cam = cv2.VideoCapture(0)
cv2.namedWindow("obra")
width = 640
height = 480
fps_limit = 15

#if USB CAM:
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
s = StateMachine(fps_limit)
time_frame = time.clock()

while True:
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        cv2.destroyAllWindows()
        break
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    #Check
    if not ret:
        print("failed to grab frame")
        break
    if frame is None:
        continue

#    while 1/(time.clock()-time_frame) > fps_limit:
#        continue

    print(s.state.__class__.__name__ ,"FPS:" , round(1/(time.clock()-time_frame),3))
    time_frame = time.clock()
    s.run(frame)


cam.release()
cv2.destroyAllWindows()
