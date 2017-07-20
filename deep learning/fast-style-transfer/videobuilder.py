import numpy as np
import cv2
import os

# cap = cv2.VideoCapture(0)
def buildvideo(style_name, output_dir,filename,fps=30):
# Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('out_' + filename + '_' + style_name + '.avi',fourcc, fps, (856,480))

    # while(cap.isOpened()):
    os.chdir(output_dir)
    for file in os.listdir(os.getcwd()):
        if file.endswith(".jpg"):
            print(file)
            # print(os.getcwd())
        # ret, frame = cap.read()
        # if ret==True:
            frame = cv2.imread(os.getcwd() + '\\' + file,1)

            #cv2.imshow('frame',frame)

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

# buildvideo('la_muse','D:\\Python\\fast-style-transfer\\Not_the_Bees_la_muse', 'Not_the_Bees')

# fps = 29.97
# buildvideo('la_muse','D:\\Python\\fast-style-transfer\\Not_the_Bees_la_muse', 'Not_the_Bees')
# buildvideo('rain_princess','D:\\Python\\fast-style-transfer\\Not_the_Bees_rain_princess', 'Not_the_Bees')
# buildvideo('scream','D:\\Python\\fast-style-transfer\\Not_the_Bees_scream', 'Not_the_Bees')
# buildvideo('udnie','D:\\Python\\fast-style-transfer\\Not_the_Bees_udnie', 'Not_the_Bees')
# buildvideo('wave','D:\\Python\\fast-style-transfer\\Not_the_Bees_wave', 'Not_the_Bees')
# buildvideo('wreck','D:\\Python\\fast-style-transfer\\Not_the_Bees_wreck', 'Not_the_Bees')
