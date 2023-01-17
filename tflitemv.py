import mvsdk
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from threading import Thread

import platform
import os

from pygame import mixer
import time
# import winsound
# from tflite_runtime.interpreter import Interpreter


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

def main():
    
    data_folder = "models/tm/aug/"
    model1 = data_folder + "model.tflite"
    model2 = data_folder + "model_edgetpu.tflite"
#     interpreter = Interpreter(model_path=model2,num_threads=4)
    interpreter = Interpreter(model_path=model2, num_threads=4, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
#     print(input_details)

    imgSize=224
    
    input_shape = input_details[0]['shape']
    # print("shape of input: ",input_shape)
    size = input_shape[1:3]
    # print("size of image: ", size) 
    print("Model Loaded Successfully.")


    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return
    
    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]
    print(DevInfo)
    hCamera = 0
    
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
        return
    
    cap = mvsdk.CameraGetCapability(hCamera)
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
    
    mvsdk.CameraSetTriggerMode(hCamera, 0)
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)
 
    mvsdk.CameraSetFrameSpeed(hCamera, 2)
    
    sRoiReslution = mvsdk.tSdkImageResolution()
    sRoiReslution.iIndex=0xff
    sRoiReslution.iWidth=640
    sRoiReslution.iWeightFOV=640
    sRoiReslution.iHeight=480
    sRoiReslution.iHeightFOV=480
    mvsdk.CameraSetImageResolution(hCamera, sRoiReslution)
#     
#     print(mvsdk.CameraGetImageResolution(hCamera))  
# 

    mvsdk.CameraPlay(hCamera)
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
    
    camfps =200
    fps_video=0
    total_video=0
    fps_video_text="0.0"
    numFrame=0
    fps_process= 0
    total_process=0
    fps_process_text="0.0"
    text=""
    color = [(0, 0, 255),(0, 255, 0)]
    color_id=0

    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
            
            frame = cv2.resize(frame, (640, 480))
            frame_resized = cv2.resize(frame, (imgSize, imgSize))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(frame_rgb, axis=0)
            fps_start_time =time.perf_counter()
            Prediction = classify(interpreter, input_data)
            fps_end_time = time.perf_counter()
            time_diff = fps_end_time - fps_start_time
            total_process+=time_diff
            total_video+=1/time_diff
            numFrame+=1
                    
            if int(Prediction) == 1 :
                text="Good"
            else:
                text="Critical"

            if numFrame == camfps :
                fps_process =  numFrame / total_process
                fps_process_text = "{:.1f}".format(fps_process)
                fps_video = total_video/numFrame
                fps_video_text = "{:.1f}".format(fps_video)
                total_video=0
                numFrame=0
                total_process=0
            print(Prediction, text, "%.4fs" % time_diff)
            cv2.putText(frame,"FPS Video: "+fps_video_text,(5,25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(frame,"FPS Processing: "+fps_process_text,(5,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(frame,text,(5,75), cv2.FONT_HERSHEY_COMPLEX, 1, color[int(Prediction)], 1)
            cv2.imshow('Demo',frame)
            
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
            

    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
