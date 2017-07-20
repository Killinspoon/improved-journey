import os

def soundify(in_dir, out_dir, audio):
    for filename in os.listdir(in_dir):
        print(filename)
        if (filename.endswith('.avi')): #or .avi, .mpeg, whatever.
            string = "ffmpeg -i {0} -i {1} -c copy {2}output_{3}".format(in_dir + filename,audio,out_dir,filename)
            print(string)
            os.system(string)
        else:
            continue

in_dir = 'D:\\Python\\soundadder\\before\\'
out_dir = 'D:\\Python\\soundadder\\after\\'
audio = 'D:\\Python\\soundadder\\audio\\not_the_bees_audio.mp3'
soundify(in_dir, out_dir, audio)
