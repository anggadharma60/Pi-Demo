import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from helper import *
from led import *
from cv_grab2 import *
import pygame

app = FastAPI()
templates = Jinja2Templates(directory="templates")
 
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

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

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

@app.get('/video_feed', response_class=HTMLResponse)
async def video_feed():
    return StreamingResponse(gen(cam),
                    media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        uvicorn.run(app, host='0.0.0.0', port=8000)
    except  Exception:
        videostream.stop()
        poweroff_led()
        sys.exit(0)