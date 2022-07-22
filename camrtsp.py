from anyio import current_time
import cv2
import os
from datetime import datetime, date
RTSP_URL = "rtsp://192.168.0.100:8554/unicast"
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)
img_count = 0
while True:
    img_count += 1
    if not cap.isOpened():
        print("Cannot open RTSP stream")
        break
    _, frame = cap.read()

    now = datetime.now()
    today = date.today()
    current_time = now.strftime("%H:%M:%S")
    cv2.putText(frame,text=f"time: {current_time} - image count: {img_count}",org=(0,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(0,255,0),thickness=2,lineType=cv2.LINE_4)
    cv2.imshow("RTSP",frame)
    cv2.imwrite(f"Pictures/{today}_{current_time}_{img_count}.jpg",frame)
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()
