import os
import cv2
import numpy as np
import math
import time
videofile='../source.mp4'
output='./output'

#@input_width @input_height @output_width @output_height @inner_radius @outer_redius @center_x, @center_y
def buildMap(Wd,Hd,R1,R2,Cx,Cy):
    map_x = np.zeros((Hd,Wd),np.float32)
    map_y = np.zeros((Hd,Wd),np.float32)
    for y in range(0,int(Hd-1)):
        for x in range(0,int(Wd-1)):
            r = (float(y)/float(Hd))*(R2-R1)+R1
            theta = (float(x)/float(Wd))*2.0*np.pi
            xS = Cx+r*np.sin(theta)
            yS = Cy+r*np.cos(theta)
            map_x.itemset((y,x),int(xS))
            map_y.itemset((y,x),int(yS))
    return map_x, map_y

def unwarp(img,xmap,ymap):
    return cv2.remap(img,xmap,ymap,cv2.INTER_LINEAR)

def video2pic(path,inner,outer):

    R1=inner #inner radius
    R2=outer #outer radius
    Wd=int(2.0*(R1+R2)/2*np.pi)
    Hd=R2-R1
    w=0
    h=0
    capture=cv2.VideoCapture(path)
    fps=15
    videoWriter = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (Wd,Hd))
    fcount=0
    if capture.isOpened():
        rval,img=capture.read()
        w,h,_=img.shape
        xmap,ymap = buildMap(Wd,Hd,R1,R2,int(w/2),int(h/2))
        videoWriter = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (Wd,Hd))
    else:
        rval=False

    while rval:
        rval,img=capture.read()
        if rval:
            matrix=cv2.getRotationMatrix2D((int(w/2),int(h/2)),45,1) #counter-clockwise rotate for 45
            img = cv2.warpAffine(img,matrix,(w,h))
            be=time.time()
            img = unwarp(img,xmap,ymap)
            img= cv2.flip(img,flipCode=-1)
            #img=cv2.resize(img,(int(Wd*0.5),int(Hd*0.5)))
            print(time.time()-be,img.shape)
            #cv2.imshow("",img)
            #cv2.waitKey()
            #cv2.imwrite('./data/'+str(fcount).zfill(6)+'_w.jpg',img)
            videoWriter.write(img)
            fcount+=1
            #cv2.waitKey(1)
    capture.release()
    videoWriter.release()

if __name__ == '__main__':
    video2pic(videofile,400,840)# manully selected based on video resolution
