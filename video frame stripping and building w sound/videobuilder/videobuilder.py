import numpy as np
import cv2
import os

# cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 23.9, (720,720))

# while(cap.isOpened()):
os.chdir("D:\\Python\\videobuilder\\input")
for file in os.listdir(os.getcwd()):
    if file.endswith(".jpg"):
        print(file)
        # print(os.getcwd())
    # ret, frame = cap.read()
    # if ret==True:
        frame = cv2.imread(os.getcwd() + '\\' + file,1)

        cv2.imshow('frame',frame)

        #
        out.write(frame)
        cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # else:
    #     break

# Release everything if job is finished
# cap.release()
out.release()
cv2.destroyAllWindows()
