import os
import cv2
import numpy as np
import math
import time

videofile='../source.mp4'
imfile='./data'

def video2pic(path):
    capture=cv2.VideoCapture(path)
    fcount=0
    if capture.isOpened():
        rval,frame=capture.read()
    else:
        rval=False
    while rval:
        rval,frame=capture.read()
        if rval:
            cv2.imwrite('./data/'+str(fcount).zfill(6)+'.jpg',frame)
            fcount+=1
            #cv2.waitKey(1)
    capture.release()

#@output_width @output_height @inner_radius @outer_redius @center_x, @center_y
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

def getCircle(img):
    w,h,c=img.shape #1728,1728,3
    print(w,h,c)
    center=(int(w/2),int(w/2))#[864,864]
    r=int(w/2) #864
    outer=int(w/2)-24# 840
    inner=400
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.circle(img,center,radius=r,color=(0,0,255),thickness=5,lineType=8,shift=0)# circle
    matrix=cv2.getRotationMatrix2D(center,47,1) #counter-clockwise rotate for 45
    img = cv2.warpAffine(img,matrix,(w,h))
    delta=0 #init degree
    #outer
    for i in range(0,4):
        img=cv2.ellipse(img,center,(outer,outer),angle=-delta,startAngle=0,endAngle=90,color=(255,0,0),thickness=5*(i%2),lineType=8,shift=0)
        delta+=90
    #inner
    for i in range(0,4):
        img=cv2.ellipse(img,center,(inner,inner),angle=-delta,startAngle=0,endAngle=90,color=(0,255,0),thickness=5*(i%2),lineType=8,shift=0)
        delta+=90
    #out=single_fisheye(img,center,outer,inner)
    t1=time.time()
    out=dewarpper(img,center,inner,outer)
    print(time.time()-t1)
    img=cv2.resize(img,(int(w*0.5),int(h*0.5)))
    out=cv2.resize(out,(int(out.shape[1]*0.5),int(out.shape[0]*0.5)))
    cv2.imshow("circle",img)
    cv2.imshow("out",out)
    cv2.waitKey()

def unwarp(img,xmap,ymap):
    output = cv2.remap(img,xmap,ymap,cv2.INTER_LINEAR)
    return output

def dewarpper(img,center,inner,outer):
    R1=inner #inner radius
    R2=outer #outer radius
    Wd=int(2.0*(R1+R2)/2*np.pi)
    Hd=R2-R1
    Ws,Hs,_=img.shape
    t1=time.time()
    xmap,ymap = buildMap(Wd,Hd,R1,R2,center[0],center[1])
    print("build map:", time.time()-t1)
    result = unwarp(img,xmap,ymap)
    return result

def improcess(path):
    imlist=os.listdir(path)
    for im in imlist:
        #print(os.path.join(path,im))
        img=cv2.imread(os.path.join(path,im))
        getCircle(img)
        #cv2.imshow("source",img)
        #cv2.waitKey()
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    #video2pic(videofile)
    improcess(imfile)
