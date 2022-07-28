import cv2;
import math;
import numpy as np;
from time import time

def DarkChannel(im,sz,pyramid=False):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);

    if pyramid:
        dc = cv2.pyrDown(dc)    

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    
    if pyramid:
        dark = cv2.pyrUp(dark)

    return dark

def AtmLight(im,dark,pyramid=False):
    if pyramid:
        im = cv2.pyrDown(im)
        dark = cv2.pyrDown(dark)

    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz,pyramid=False):
    if pyramid:
        im = cv2.pyrDown(im)

    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);
    
    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    print(transmission)

    if pyramid:
        transmission = cv2.pyrUp(transmission)
    
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et, pyramid=False):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    r = 60;
    eps = 0.0001;
    if pyramid:
        gray = cv2.pyrDown(gray)
        et = cv2.pyrDown(et)

    t = Guidedfilter(gray,et,r,eps);
    
    if pyramid:
        t = cv2.pyrUp(t)
    return t;

# transmission light boost
def P(t):
    if 0<t<0.5:
        return 2*t
    else: 
        return 1

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    dim_A = (int(im.shape[1]),int(im.shape[0]))
    A = cv2.resize(A, dim_A, interpolation = cv2.INTER_AREA)
    t = cv2.resize(t, dim_A, interpolation = cv2.INTER_AREA)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/(t) + A[0,ind]

    return res

def dehaze(I, pyramid=False):
    I = np.float32(I)
    # time0 = time()
    dark = DarkChannel(I,5,pyramid=pyramid)
    # dark = cv2.medianBlur(dark,3)
    # time1 = time()
    # print("DarkChannel time: %.2f" % (time1-time0))

    A = AtmLight(I,dark,pyramid=pyramid)
    # time2 = time()
    # print("Atmlight time: %.2f" % (time2-time1))
 
    te = TransmissionEstimate(I,A,5,pyramid=pyramid)
    # time3 = time()
    # print("Transmission estimate time: %.2f" % (time3-time2))

    # t = TransmissionRefine(I,te,pyramid=pyramid)
    # t = cv2.medianBlur(np.float32(te),3)
    # time4 = time()
    # print("Transmission Refine time: %.2f" % (time4-time3))

    J = Recover(I,te,A,0.1)
    # time5 = time()
    # print("Recovery time: %.2f" % (time5-time4))

    J[J>255] = 255
    J[J<0] = 0
    
    return J

def lowlight_enhance(src,pyramid=False, light_boost = False):
    # start = time()
    src = 255-src
    src = dehaze(src,pyramid=pyramid)
    src = 255-src
    # print(f"Enhance time: %.2f" % (time()-start))
    return src

if __name__ == '__main__':
    
    # img = cv2.imread('images/fog_01.png')
    # img = cv2.imread('images/cat.jpg')
    # img = cv2.imread('images/Car/2015_02409.jpg')
    img = cv2.imread('images/car_improved_dcp.png')
    start = time()
    img = lowlight_enhance(img,pyramid=True,light_boost=False)
    end = time()
    print(end-start)
    # img = dehaze(img, pyramid=True)
    # cv2.imwrite("./dehaze_result_pyramid.png",img)
    cv2.imwrite("./result_lle_dcp.png",img)
    # img = cv2.imread("")


    