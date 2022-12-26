from flask import Flask, Response, render_template, request, send_file
from helper import *
from led import *
from cv_grab2 import *
import pygame

app = Flask(__name__)
 
interpreter = load_interpreter(list_model)
camfps=200
DevList = mvsdk.CameraEnumerateDevice()
nDev = len(DevList)
if nDev < 1:
    print("No camera was found!")

for i, DevInfo in enumerate(DevList):
    print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
i = 0 if nDev == 1 else int(input("Select camera: "))
DevInfo = DevList[i]

cam = Camera(DevInfo)
cam.open()

# time.sleep(1.0)


@app.route('/')
def index():
    return render_template('index.html')

def gen(cam):
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
        frame = cam.grab()
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
    return Response(gen(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', debug=False)
    except Exception:
        cam.close()
        poweroff_led()
        sys.exit(0)
    