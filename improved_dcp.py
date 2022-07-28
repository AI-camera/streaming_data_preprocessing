import cv2
import time
import math
import numpy as np

def DarkChannel(im,sz,pyramid=False):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);

    if pyramid:
        dc = cv2.pyrDown(dc)    

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    # dark = cv2.erode(dc,kernel)
    dark = cv2.medianBlur(dc,kernel)

    dark = dc

    if pyramid:
        dark = cv2.pyrUp(dark)
    return dark

def AtmLight(im,dark,pyramid=False):
    if pyramid:
        im = cv2.pyrDown(im)
        dark = cv2.pyrDown(dark)

    [h,w] = im.shape[:2]
    imsz = h*w
    # numpx = int(max(math.floor(imsz/1000),1))
    numpx = 16
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def Recover(im,A,t=0.5,w=1):
    res = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + w*A[0,ind]

    return res

def dark_coarse_loose(img,A):
    b,g,r = cv2.split(img)
    b = b/A[0][0]
    g = g/A[0][1]
    r = r/A[0][2]
    dc = cv2.min(cv2.min(r,g),b);
    return dc.astype("float32")

def dehaze(image,pyramid=False):
    image= np.float32(image)
    image_down = cv2.pyrDown(image)

    dark_coarse = DarkChannel(image_down,5,pyramid=pyramid)
    A = AtmLight(image_down, dark_coarse, pyramid = pyramid)
    dark_coarse = dark_coarse_loose(image_down,A)

    med = cv2.medianBlur(dark_coarse, 5)
    detail = cv2.absdiff(med, dark_coarse)
    detail = cv2.medianBlur(detail, 5)

    smooth = med - detail

    dark_coarse = dark_coarse*0.98
    dark = cv2.min(dark_coarse, smooth)
    t = 1.0 - dark*0.95

    A = AtmLight(image_down, dark, pyramid = pyramid)
    t = cv2.pyrUp(t)

    t = cv2.resize(t, (image.shape[1],image.shape[0]))

    A = cv2.pyrUp(A)
    image = Recover(image, A,t)
    return image

def dehaze_modified(image,pyramid=False):
    image_down = cv2.pyrDown(image)

    dark_coarse = DarkChannel(image_down,5,pyramid=pyramid)
    A = AtmLight(image_down, dark_coarse, pyramid = pyramid)
    A = cv2.pyrUp(A)

    image = Recover(image, A,t=0.5,w=0.8)
    return image

def lowlight_enhance(image,pyramid=False):
    image = 255 - image
    image = dehaze(image,pyramid=pyramid)
    image = 255 - image
    return image

def lowlight_enhance_modified(image,pyramid=False):
    image = 255 - image
    image = dehaze_modified(image,pyramid=pyramid)
    image = 255 - image
    return image

if __name__ == '__main__':
    # img = cv2.imread('images/Car/2015_02409.jpg')
    # img = cv2.imread('images/car_improved_dcp.png')
    img = cv2.imread('images/eval15/low/1.png')
    start = time.time()
    img = lowlight_enhance(img,pyramid=False)
    # img = lowlight_enhance_modified(img,pyramid=True)
    end = time.time()
    print(end-start)
    cv2.imwrite("./result_lle_improved.png",img)
