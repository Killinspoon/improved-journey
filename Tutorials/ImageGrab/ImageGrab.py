import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, UP, DOWN, LEFT, RIGHT, START, SELECT, L, R

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
    except:
        pass

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1=200, threshold2=300)
    processed_image = cv2.GaussianBlur(processed_image, (5,5), 0)
    vertices = np.array([[0,370], [60,135],[420,135], [480,370]])
    # vertices = np.array([[370,0], [135,0], [135,480], [370,480]])
    processed_image = roi(processed_image,[vertices])

    #                      edges
    lines = cv2.HoughLinesP(processed_image, 1, np.pi/180, 180, 2, 15)
    draw_lines(processed_image, lines)

    return processed_image


# def main():
    # Not required as we aren't catching input
    # for i in list(range(4))[::-1]:
    #     print(i+1)
    #     time.sleep(1)

last_time = time.time()
while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,50,480,370)))
    new_screen = process_img(screen)
    print('Loop took {} seconds'.format(time.time() - last_time))
    last_time = time.time()

    cv2.imshow('window2',new_screen)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
