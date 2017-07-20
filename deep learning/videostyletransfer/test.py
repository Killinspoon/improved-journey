import os
import cv2
from PIL import Image, ImageChops, ImageOps


# Takes an input and an output image and makes them square... (like the shitty videos you see on Facebook)
def makeImageSquare(f_in, size=None):

    image = Image.open(f_in);

    if size == None:
        width, height = image.size;
        dim = min(width, height); # As in, the dimension of the square result, made nice for EJs GPU ;)
        size = (dim, dim);

    image.thumbnail(size, Image.ANTIALIAS);
    image_size = image.size;

    thumb = image.crop( (0, 0, size[0], size[1]) )
    offset_x = int(max( (size[0] - image_size[0]) / 2, 0 ));
    offset_y = int(max( (size[1] - image_size[1]) / 2, 0 ));

    thumb = ImageChops.offset(thumb, offset_x, offset_y);
    thumb.save(f_in);


def stripVideoFrames(f_in, folder):
    vidcap = cv2.VideoCapture(f_in)
    count = 0
    while True:
        success,image = vidcap.read()
        # squareImage = makeImageSquare(image)
        # squareImage.save(folder + str(count) + ".jpg");

        if not success:
            break
        # print(folder + "\\{:04d}_frame.jpg".format(count))
        cv2.imwrite(folder + "\\{:04d}_frame.jpg".format(count), image)     # save frame as JPEG file
        # print(folder + "\\{:04d}_frame.jpg".format(count))
        makeImageSquare(folder + "\\{:04d}_frame.jpg".format(count))
        count += 1
    print("{} images are extacted in {}.".format(count,folder))


# BUILD OUTPUT DIRECTORY IF IT DOESN'T EXIST
os.chdir("D:\\Python\\videostyle")
folder = 'test_output'
if not os.path.isdir(folder):
    os.mkdir(folder)

video_file = 'sample.mp4'
output_dir = "\\" + folder + "\\"

print(video_file)
print(output_dir)

stripVideoFrames(video_file, folder)

#
#
# # DO THE THING WOOOO
# source = "D:\\Python\\opencv2\\test\\frame0.JPG"
# output = "D:\\Python\\opencv2\\test_pad\\image_padded.JPG"
#
# makeImageSquare(source)
