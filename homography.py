import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.insert(0,'/usr/local/lib/python2.7/site-packages/')
import cv2
print cv2.__version__
scale = 1
delta = 0
ddepth = cv2.CV_16S
# needed for orb because of a bug
cv2.ocl.setUseOpenCL(False)

def getInteriorPoints(img,poly):
    nx,ny,nc = img.shape
    x = np.linspace(0,nx,nx)
    y = np.linspace(0,ny,ny)
   
    intList = []
    img2 = img.copy()
    cList = []
    for i in range(4):
        xi = 2*i
        yi = 2*i + 1
        xI = poly[xi] 
        yI = poly[yi]
        cList.append((yI,xI))
    
    
    contour = np.array( cList, np.int )
    
    #cv2.drawContours(img, [contour], 0, (0, 255, 0), 3) 
    #plt.figure()
    #plt.imshow(img)
    #plt.show()
    b = [0,0,255] # blue for testing
    for i in x:
        for j in y:
            pt =  (i,j)
            if cv2.pointPolygonTest(contour,pt,False) == 1:
                intList.append((i,j))
                for c in range(3):
                    img2[int(i),int(j),c] = b[c]
    plt.imshow(img2)
    plt.scatter(contour[:2,1],contour[:2,0])
    plt.figure()
    
    #plt.figure()            
    return intList

def estHomography(video_pts, logo_pts):
    # video_pts is a 4x2 matrix of points of the corners of the goal
    # logo_pts is a 4x2 matrix with the corners of the clock
    # A will be an 8x 9 matrix with rows alternating between
    # ax and ay
    A = []
    r,c = video_pts.shape
    for i in range(r):
        x1 = video_pts[i,1]
        y1 = video_pts[i,0]
        x1p = logo_pts[i,0]
        y1p = logo_pts[i,1]
        
        ax = [-x1,-y1,-1,0,0,0,x1*x1p,y1*x1p, x1p]
        ay = [0,0,0,-x1,-y1,-1,x1*y1p, y1*y1p, y1p]
        A.append(ax)
        A.append(ay)
    Am = np.array(A)
    Am = np.reshape(Am,(8,9))
    U,S,VT = np.linalg.svd(Am)
    V = VT.T
    eV1 = V[:,-1]   # select vector with smallest eigenvalue
    H = np.reshape(eV1,(3,3))
   
    for i in range(r):
        x1 = video_pts[i,1]
        y1 = video_pts[i,0]
        r = np.array([x1,y1,1])
        plogo = np.dot(H,r.T)
        lam = plogo[2]
        #print plogo
        print "orig ", x1,y1,  "calc ", plogo/lam
        
    return H
    
def warp_pts( video_pts, logo_pts, sample_pts):
    
    H1 = estHomography(video_pts,logo_pts)
    wList = []
    ns, nc = sample_pts.shape
    print " min sample ", np.min(sample_pts,axis=0), " max ", np.max(sample_pts,axis=0)
    for p in range(ns):
        x1 = sample_pts[p,0]
        y1 = sample_pts[p,1]
        r = np.array([x1, y1, 1.])
        r = np.reshape(r,(3,1))
        #print r
        rp = np.dot(H1,r)
        lam = rp[2]
        #print
        #print " lambda " , lam
        xw = rp[0]/lam
        yw = rp[1]/lam
        rw = [xw,yw]
        #print " map ", x1, y1, " to ", xw,yw
        wList.append(rw)
   
    #print "  Homography ", H1
    wMat = np.array(wList)
    wMat = np.squeeze(wMat)
    print " min ", np.min(wMat,axis=0), " max ", np.max(wMat,axis=0)
  
    
    return wMat

def inverse_warping(img, imgLogo,sample_pts, warped_pts,H):
    imgF = img.copy()
    xl,yl,cl = imgLogo.shape
    ns, nc = sample_pts.shape
    print " num points ", ns, nc
    print " range sample ", np.min(sample_pts,axis=0), np.max(sample_pts,axis=0)
    print " range warped", np.min(warped_pts,axis=0), np.max(warped_pts,axis=0)
    
    for i in range(ns):
        x = int(sample_pts[i,0])
        y = int(sample_pts[i,1])
        r = np.array([x,y,1])
        rp = np.dot(H,r)
        #xp = abs(int(math.floor(warped_pts[i,0])))
        #yp = abs(int(math.floor(warped_pts[i,1])))
        xp = abs(int(rp[0]/rp[2]))
        yp = abs(int(rp[1]/rp[2]))
        #print xp,yp
        if xp > 399:
            xp = 399
        if yp > 989:
            yp = 989
        #print x,y,xp,yp
        b = [0,0,255] # blue for testing
        for c in range(3):
            imgF[x,y,c] =  imgLogo[xp,yp,c]
    return imgF
    
    return []
  
def appendImages(img1,img2): # need revision
    row1 = img1.shape[0]
    row2 = img2.shape[0]
    
    if row1 < row2:
        img1 = np.concatenate((img1,np.zeros((row2-row1,img1.shape[1]))),axis=0)
    elif row1 >row2:
        img2 = np.concatenate((img2,np.zeros((row1-row2,img2.shape[1]))),axis=0)
    
    return np.concatenate((img1,img2),axis=1)
    