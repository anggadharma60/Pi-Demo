from helper import *
from led import *
def main():
    
    data_folder = "models/"
    model1 = data_folder + "MobileNetV2_V2.tflite"
    model2 = data_folder + "MobileNetV2_V2_edgetpu.tflite"
    list_model=[model1, model2]
    interpreter = load_interpreter(list_model)

    print("Model Loaded Successfully.")
    imgSize=224
    camfps=30
    videostream = WebcamVideoStream(src=0,
                                    res=(640, 480), fps=camfps, api = 0).start()
    
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
    while True:
        
        _,frame = videostream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (imgSize, imgSize))
        input_data = np.expand_dims(frame_resized, axis=0)
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
        if numFrame >= camfps :
            fps_process =  numFrame / total_process
            fps_process_text = "{:.1f}".format(fps_process)
            fps_video = total_video/numFrame
            fps_video_text = "{:.1f}".format(fps_video)
            total_video=0
            numFrame=0
            total_process=0
        
#         print(Prediction, text, "%.4fs" % time_diff)
        show_led(Prediction)
        cv2.putText(frame,"FPS Video: "+fps_video_text,(5,25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(frame,"FPS Processing: "+fps_process_text,(5,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(frame,text,(5,75), cv2.FONT_HERSHEY_COMPLEX, 1, color[int(Prediction)], 1)
        cv2.imshow('Demo',frame)
        hide_led()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    videostream.stop()
    cv2.destroyAllWindows()
    poweroff_led()


if __name__ == '__main__':
    main()