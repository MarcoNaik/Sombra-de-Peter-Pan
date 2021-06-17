import recognizer
import face_swapping_1
import time
import threading
import cv2

from utilities import *

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
def new_vid(fps):
    return cv2.VideoWriter('registro/registro-{}.mp4'.format(str(time.time())),fourcc, fps, (256,256))


class Start:
    def __init__(self , statemachine):
        self.statemachine = statemachine
        self.black = cv2.imread("black.png")

    def next_state(self, cropped_face_cords):
        self.statemachine.to_default(cropped_face_cords)

    def run(self, frame):
        display = self.black
        cv2.imshow("obra", display)
        self.statemachine.current_video.write(display)
        face = detectFaces(frame)
        if face is not None:
            cropped_face_cords = broaderFaceCords(face[0])
            if cropped_face_cords is not None:
                if not is_any_negative(cropped_face_cords):
                    self.next_state(cropped_face_cords)


class Default:
    def __init__(self, statemachine, driving_video, cropped_face_cords):
        self.statemachine = statemachine
        self.driving_video = driving_video
        self.cropped_face_cords = cropped_face_cords
        self.t0 = time.clock()
        self.t1 = time.clock()
        self.delay_time = 2

    def FOMwaiting(self, current_driving_video, statemachine):
        FOM_thread = threading.Thread(target = runFOM,args=(current_driving_video,), daemon = True)
        FOM_thread.start()
        FOM_thread.join()

        NNcap = cv2.VideoCapture("result.mp4")
        statemachine.to_end(NNcap, self.cropped_face_cords)

    def next_state(self):
        self.statemachine.to_transition(self.cropped_face_cords)

    def run(self, frame):

        new_cropped_face_cords =new_cropped_face_cords_func(frame, self)
        if new_cropped_face_cords is None:
            self.t0 = time.clock()
            self.t1 = time.clock()
            return

        new_cropped_face_cords, face, gray = new_cropped_face_cords

        if (self.t1 - self.t0) < self.delay_time:
            self.t1 = time.clock()
            show_previous_cropped_frame(frame, self)
            return

        cropped_frame = crop(frame, self.cropped_face_cords)
        #cv2.imshow("obra", cropped_frame)
        #self.statemachine.current_video.write(cropped_frame)

        writePhoto(cropped_frame)
        #face_swapping_1.set_up()
        self.next_state()
        #Run NN
        thread = threading.Thread(target = self.FOMwaiting, args=(self.driving_video, self.statemachine,), daemon = True)
        thread.start()


class Transition:
    def __init__(self , statemachine, cropped_face_cords):
        self.statemachine = statemachine
        self.cropped_face_cords = cropped_face_cords
        self.t0 = time.clock()
        self.t1 = time.clock()
        self.t2 = 0
        self.wait_time = 10
        self.blending_time = 10
        self.swapper = face_swapping_1.Face_Swapper()

    def run(self, frame):
        new_cropped_face_cords =new_cropped_face_cords_func(frame, self)
        if new_cropped_face_cords is None:
            return

        new_cropped_face_cords, face, gray = new_cropped_face_cords
        if (self.t1 - self.t0) < self.wait_time:
            self.t1 = time.clock()
            self.t2 = self.t1
            show_previous_cropped_frame(frame, self)
            return

        swappedframe = self.swapper.face_swap(frame, face, gray)
        if swappedframe is None:
            show_previous_cropped_frame(frame, self)
            return
        cropped_face_swapped = crop(swappedframe, self.cropped_face_cords)
        #crossfade:
        crossfrade_delta = self.t2 - self.t1
        if (crossfrade_delta) < self.blending_time:
            self.t2 = time.clock()

            alpha = crossfrade_delta/self.blending_time
            beta = (1.0 - alpha)

            cropped_frame = crop(frame, self.cropped_face_cords)

            blend = cv2.addWeighted(cropped_face_swapped, alpha, cropped_frame , beta, 0.0)
            cv2.imshow("obra", blend)
            self.statemachine.current_video.write(blend)
        else:
            cv2.imshow("obra", cropped_face_swapped)
            self.statemachine.current_video.write(cropped_face_swapped)
        pass

class End:
    def __init__(self, statemachine, NNcap, cropped_face_cords):
        self.statemachine = statemachine
        self.NNcap = NNcap
        self.t0 = time.clock()
        self.t1 = time.clock()
        self.t2 = time.clock()
        self.blending_time = 5
        self.exit_time = 3
        self.swapper = face_swapping_1.Face_Swapper()
        self.cropped_face_cords = cropped_face_cords

    def run(self, frame):
        NN_, NNimg = self.NNcap.read()
        if NNimg is None:
            self.NNcap = cv2.VideoCapture("result.mp4")
            NN_, NNimg = self.NNcap.read()

        crossfrade_delta = self.t1 - self.t0
        if crossfrade_delta < self.blending_time:
            self.t1 = time.clock()

            new_cropped_face_cords =new_cropped_face_cords_func(frame, self)
            if new_cropped_face_cords is None:
                return

            new_cropped_face_cords, face, gray = new_cropped_face_cords
            swappedframe = self.swapper.face_swap(frame, face, gray)
            if swappedframe is None:
                show_previous_cropped_frame(frame, self)
                return
            cropped_face_swapped = crop(swappedframe, self.cropped_face_cords)

            alpha = crossfrade_delta/self.blending_time
            beta = (1.0 - alpha)

            blend = cv2.addWeighted(NNimg, alpha, cropped_face_swapped , beta, 0.0)
            cv2.imshow("obra", blend)
            self.statemachine.current_video.write(blend)
            self.t2 = time.clock()
        else:
            faces = detectFaces(frame)

            if faces is None:
                if time.clock() - self.t2  > self.exit_time:
                    self.statemachine.restart()
            else:
                self.t2 = time.clock()

            cv2.imshow("obra", NNimg)
            self.statemachine.current_video.write(NNimg)

class StateMachine():
    def __init__(self, fps):
        self.state = Start(self)
        self.fps = fps
        self.current_video = new_vid(self.fps)

    def current_state(self):
        return self.state

    def to_default(self, cropped_face_cords):
        current_driving_video = "driving_video/" + get_driving_video()
        self.state = Default(self, current_driving_video, cropped_face_cords)
    def to_transition(self, cropped_face_cords):
        self.state = Transition(self, cropped_face_cords)
    def to_end(self, NNcap, cropped_face_cords):
        self.state = End(self, NNcap, cropped_face_cords)
    def restart(self):
        self.current_video.release()
        self.current_video = new_vid(self.fps)
        self.state = Start(self)

    def run(self, frame):
        self.state.run(frame)
