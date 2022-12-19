import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from threading import Thread
import os
import platform
from pygame import mixer
import time

def check_platform():
    return platform.system()

def load_interpreter(list_model):  
    platform = check_platform()
    edgetpu= None
    try:
        if platform == "Windows":
            edgetpu = load_delegate('edgetpu.dll')
        if platform == "Linux":
            edgetpu = load_delegate('libedgetpu.so.1', options={"device": ":0"})
    except:
        pass
    if edgetpu is None:
        print("Edge TPU Not Detected")
        interpreter = Interpreter(model_path=list_model[0], num_threads=4)
    else:
        print("Edge TPU Detected")
        interpreter = Interpreter(model_path=list_model[1], num_threads=4,
                                      experimental_delegates=[edgetpu])
    interpreter.allocate_tensors()
    return interpreter


        
def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    scale, zero_point = input_details['quantization']
    input_tensor[:, :] = np.uint8(image / scale + zero_point)


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def classify(interpreter, image):
    output_details = interpreter.get_output_details()
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    scale, zero_point = output_details[0]['quantization']
    output = get_output_tensor(interpreter, 0)
    output = scale * (output - zero_point)
    top_1 = np.argmax(output)
    return top_1

class WebcamVideoStream:
    def __init__(self, src=0, res=(640,480), fps=30, api=0,name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src, api)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        (self.grabbed, self.frame) = self.stream.read()
  
        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return True,self.frame
    
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True