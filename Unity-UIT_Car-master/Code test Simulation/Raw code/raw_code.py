import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import cv2
from PIL import Image
from flask import Flask
from io import BytesIO
#------------- Add library ------------#
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
    #x1, y1, x2, y2 in lines
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 0, 255), thickness=4)

    img = cv2.addWeighted(img, 0.4, blank_image, 1, 0.0) #0.8
    return img

def nothing(x):
    pass
#--------------------------------------#
#Global variable
MAX_SPEED = 30
MAX_ANGLE = 25
# Tốc độ thời điểm ban đầu
speed_limit = MAX_SPEED
MIN_SPEED = 10

#intialize
#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        steering_angle = 0  #Góc lái hiện tại của xe
        speed = 0           #Vận tốc hiện tại của xe
        image = 0           #Ảnh gốc

        steering_angle = float(data["steering_angle"])
        speed = float(data["speed"])
        print(steering_angle, speed)
        #Original Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ######## them ne
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = np.ones((5,5), np.float32)/25
        image1 = cv2.bilateralFilter(image1, 9, 75, 75) #lọc các nhiễu rìa ảnh
        image1 = cv2.filter2D(image1, -1, kernel) #lọc ảnh theo kiểu làm mờ
        height = image1.shape[0]
        width = image1.shape[1]
        region_of_interest_vertices = [
            (0, height - 40),
            (width/2 - 10, height/2 - 10),
            (width, height - 40)
                ]
        gray_image = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        canny_image = cv2.Canny(gray_image, 100, 150)
        cropped_image = region_of_interest(canny_image,
                np.array([region_of_interest_vertices], np.int32),)
        ################################################
        #################################################
        region_of_interest_vertices_1 = [
            (width, height),
            (0, height),
            (0, 2*height/3),
            (width/2, 2*height/3 - 10),
            (width, 2*height/3)
                ]
        cropped_image_1 = region_of_interest(canny_image, 
                np.array([region_of_interest_vertices_1], np.int32),)
        ################################################3
        lines = cv2.HoughLinesP(cropped_image_1,
                        rho=1,
                        theta=np.pi/180,
                        threshold=0,
                        lines=np.array([]),
                        minLineLength=0,
                        maxLineGap=0)

        image_with_lines = drow_the_lines(image, lines)
        ###################
        """
        - Chương trình đưa cho bạn 3 giá trị đầu vào:
            * steering_angle: góc lái hiện tại của xe
            * speed: Tốc độ hiện tại của xe
            * image: hình ảnh trả về từ xe
        
        - Bạn phải dựa vào 3 giá trị đầu vào này để tính toán và gửi lại góc lái và tốc độ xe cho phần mềm mô phỏng:
            * Lệnh điều khiển: send_control(sendBack_angle, sendBack_Speed)
            Trong đó:
                + sendBack_angle (góc điều khiển): [-25, 25]  NOTE: ( âm là góc trái, dương là góc phải)
                + sendBack_Speed (tốc độ điều khiển): [-150, 150] NOTE: (âm là lùi, dương là tiến)
        """
        sendBack_angle = 0
        sendBack_Speed = 0
        try:
            #------------------------------------------  Work space  ----------------------------------------------#

            cv2.imshow("Line", image_with_lines)
            #cv2.imshow("Canny", canny_image)
            #cv2.imshow("Cropped", cropped_image_1)
            #cv2.imshow('sobely', canny_image_1)

            cv2.waitKey(1)
            #------------------------------------------------------------------------------------------------------#
            print('{} : {}'.format(sendBack_angle, sendBack_Speed))
            send_control(sendBack_angle, sendBack_Speed)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)


if __name__ == '__main__':
    
    #-----------------------------------  Setup  ------------------------------------------#


    #--------------------------------------------------------------------------------------#
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    
