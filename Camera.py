import cv2,pandas
from datetime import datetime
from image_recognition import predict
import os

def save(file_name,frame):
    cv2.imwrite(filename=file_name, img=frame)
    # img_new = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # img_new = cv2.imshow("Saved Image", img_new)
    # os.save(file_name)
    predict(file_name)
    os.remove(file_name)
def take():
    first_frame = None

    video = cv2.VideoCapture(0)

    while True:
        check,frame = video.read()

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(21,21),0)

        if first_frame is None:
            first_frame = gray
            continue
        delta_frame = cv2.absdiff(first_frame,gray)
        thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

        # cv2.imshow("Threshold Frame",thresh_frame)
        cv2.imshow("Color Frame",frame)

        key=cv2.waitKey(1)
        if key == ord('s'):
            # s = input("Enter the File Name : ")
            save('img.jpg',frame)
            break

        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows

take()
# os.system("img.jpg")
