from flask import Flask, Response, render_template, request, send_file
from helper import *
from led import *
import pygame
import sys

app = Flask(__name__)
 
data_folder = "models/"
model1 = data_folder + "MobileNetV2_V2.tflite"
model2 = data_folder + "MobileNetV2_V2_edgetpu.tflite"
# model1 = data_folder + "model.tflite"
# model2 = data_folder + "model_edgetpu.tflite"
list_model=[model1, model2]
critical_sound = 'templates/critical.mp3'
good_sound = 'templates/good.mp3'
list_sound = [critical_sound, good_sound]
interpreter = load_interpreter(list_model)

imgSize=224
camfps=30
videostream = WebcamVideoStream(src=0, res=(480, 360), fps=camfps, api=cv2.CAP_V4L).start()
# time.sleep(1.0)


@app.route('/')
def index():
    return render_template('index.html')

def gen(videostream):
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
    sound=""
    while True:
        fps_start_time =time.perf_counter()
        _,frame = videostream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (imgSize, imgSize))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        Prediction = classify(interpreter, input_data)
    
        if int(Prediction) == 1 :
            text="Good"
            sound=list_sound[int(Prediction)]
            pygame.mixer.init()
            pygame.mixer.music.load(sound)
        else:
            text="Critical"
            sound=list_sound[int(Prediction)]
            pygame.mixer.init()
            pygame.mixer.music.load(sound)
        
        cv2.putText(frame,"FPS Video: "+fps_video_text,(5,25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(frame,"FPS Processing: "+fps_process_text,(5,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(frame,text,(5,75), cv2.FONT_HERSHEY_COMPLEX, 1, color[int(Prediction)], 1)
        fps_end_time = time.perf_counter()
        time_diff = fps_end_time - fps_start_time
        total_process+=time_diff
        total_video+=1/time_diff
        numFrame+=1

        if numFrame == camfps :
            fps_process =  numFrame / total_process
            fps_process_text = "{:.1f}".format(fps_process)
            fps_video = total_video/numFrame
            fps_video_text = "{:.1f}".format(fps_video)
            total_video=0
            numFrame=0
            total_process=0
        
        ret , jpg = cv2.imencode('.jpg', frame)
        jpg = jpg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n\r\n')
        show_led(Prediction)
        pygame.mixer.music.play()
        hide_led()
        
@app.route('/video_feed')
def video_feed():
    return Response(gen(videostream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', debug=False)
    except Exception:
        videostream.stop()
        poweroff_led()
        sys.exit(0)
