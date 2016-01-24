#import pcl
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

sift = cv2.SIFT()
show_rt = True
DEBUGGING = False
UNIT_TESTING = False
CAM_TEST = True
MEM_TEST = False
CAM_READ = False
CR = True
c_no = 1
cd = 1
cam_rd_ct = 30

f = 4 #focal length of logitech 310, in mm
sensor_size_x = 3.6     #in mm
sensor_size_y = 2.7     #in mm
sensor_size = sensor_size_x * sensor_size_y     #in mm^2

def compute_P(K, R, T):
    #calculate intrinsics matrix
    M_int = K
    M_int = np.c_[M_int, [0, 0, 0]]

    #calculate extrinsics
    M_ext = R
    M_ext = np.c_[M_ext, T]
    M_ext = np.r_[M_ext, [[0, 0, 0, 1]]]

    #calculate projection
    P = np.dot(M_int, M_ext)

    if DEBUGGING:
        print '\nIntrinsics Matrix:\n', M_int, '\n'
        print '\nExtrinsics Matrix:\n', M_ext, '\n'
        print '\nProjection Matrix:\n', P, '\n'

    return P

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''

    #if DEBUGGING:
     #   print '\nimg1.shape:\n', img1.shape, '\n'

    r,c, = img1.shape

    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c) / r[1]])
        cv2.line(img1, (x0, y0), (x1, y1), color,1)
        cv2.circle(img1, tuple(pt1), 5, color, -1)
        cv2.circle(img2, tuple(pt2), 5, color, -1)

    #if DEBUGGING:
     #   pass
        #print '\nimg1 in drawlines():\n', img1, '\n'
        #print '\nimg2 in drawlines():\n', img2, '\n'

    return img1, img2

def compute_K(im):
    x_pixels = len(im[0])
    y_pixels = len(im)
    total_pixels = x_pixels * y_pixels

    pixel_size_x = x_pixels / sensor_size_x   #in pixels/mm
    pixel_size_y = y_pixels / sensor_size_y     #in pixels/mm

    f_x = f * pixel_size_x     #pixels
    f_y = f * pixel_size_y     #pixels

    global f_pix
    f_pix = f * ((pixel_size_x * pixel_size_y) / 2)

    K = np.asarray([[f_x, 0, x_pixels/2],
                    [0, f_y, y_pixels/2],
                    [0, 0, 1]])

    if DEBUGGING:
        print '\nCamera Matrix:\n', K, '\n'

    return K

def init_camera(cam_no, height, width):
    #global camera
    camera = cv2.VideoCapture(cam_no)    #gets input from the ith camera, 0 in this case
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
    if DEBUGGING:
        print'\nInitialized Camera\n'
    return camera

def get_frame(camera):
    ret, frame = camera.read()  # get the current frame
    #frame = cv2.flip(frame, 1)  #use '1' to mirror, zero for upside down
    if DEBUGGING:
        print '\nGot a frame from the camera\n'

    return cv2.flip(frame, 1)

def compute_F(im1, im2):
    if show_rt:
        t = time.time()
    #get points in frame A
    #get sift points to match (kp1)
    kp1, des1 = sift.detectAndCompute(im1, None)

    #get points in frame B
    #get sift points to match (kp2)
    kp2, des2 = sift.detectAndCompute(im2, None)

    #only take points that are in both images to compute fundamental matrix
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    #cleaner, from http://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html#gsc.tab=0
    #no idea whats going on here
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.6*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    pts1 = pts1.astype(float)
    pts2 = pts2.astype(float)

    #compute fundamental matrix based on given keypoints (F)
    F, mask = cv2.findFundamentalMat(pts1.astype(float),pts2.astype(float),cv2.FM_RANSAC)
    #F, mask = cv2.findFundamentalMat(pts1, pts2, method=CV_FM_RANSAC)     ^^cv2.FM_LMEDS

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)


    if UNIT_TESTING:
        # We select only inlier points

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = drawlines(im1,im2,lines1,pts1,pts2)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        img3,img4 = drawlines(im2,im1,lines2,pts2,pts1)

        #print '\n', 'img5:', '\n', img5, '\n'
        #cv2.imshow('img5', img5)
        #cv2.imshow('img3', img3)
        #cv2.waitKey(0)

        plt.subplot(121),plt.imshow(img5)
        plt.subplot(122),plt.imshow(img3)
        plt.show()

    if DEBUGGING:
        print '\nFundamental Matrix:\n', F, '\n'

    if show_rt:
        t = time.time() - t
        print '\ncompute_F() runtime:', t, 'secs\n'

    return F, pts1, pts2

def compute_E(F,im1,im2):
    #E = Kt * F * K
    K_l = compute_K(im1)
    K_r = compute_K(im2)
    E = np.dot(F, K_r)
    E = np.dot(np.transpose(K_l), E)
    if DEBUGGING:
        print '\nEssential Matrix:\n', E, '\n'

    return E

def compute_R_T(E):

    U,S,V = np.linalg.svd(E)

    V = V.T

    if DEBUGGING:
        print "\nsvd decomp\n"
        #print U,S,V
        print '\nU:\n', U, '\n'
        print '\nS:\n', S, '\n'
        print '\nV:\n', V, '\n'

    mid = np.matrix([ [0,-1,0],
                      [1, 0,0],
                      [0, 0,1] ])

    prodA = np.dot(mid, V.T)

    rotation = np.dot(U, prodA)

    #prod1 = np.dot(U,mid)

    #rotation = np.dot(prod1,V.T)

    translation = np.transpose(np.matrix([U[0][2],U[1][2],U[2][2]]))

    if DEBUGGING:
        print '\nRotation Matrix:\n', rotation, '\n'
        print '\nTranslation Matrix:\n', translation, '\n'

    return rotation, translation

def plot_3D(im1, im2, img3D):
    return 0

    if show_rt:
        t = time.time()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for i in range(0,len(im1), 1):
        for j in range(0,len(im1[0]), 1):
            x = img3D[j][i][0]
            y = img3D[j][i][1]
            z = img3D[j][i][2]
            '''
            print '\ni:', i
            print '\nj:', j
            print '\nlen(im1):', len(im2)
            print '\nlen(im1[0]):', len(im2[0])
            '''
            #print '\nplotting, giving c:\n', im2[i,j], '\n'
            #ax.scatter(x,y,z, c=im2[i,j], marker='o')
            ax.scatter(x,y,z, c=im2[i,j], marker='o')

    ax.set_xlabel('x axis')
    ax.set_xlabel('y axis')
    ax.set_xlabel('z axis')

    #print "Made IT!"
    #plt.show()
    #cv2.waitKey(0)
    if show_rt:
        t = time.time() - t
        print '\nplot_3D runtime:', t, 'secs\n'

def get_cld(im1, im2): #im1 and im2 must be black and white
    if show_rt:
        t = time.time()

    #compute Fundamental matrix
    F, pts1, pts2 = compute_F(im1, im2)

    #compute Essential matrix
    E = compute_E(F, im1, im2)

    #get translation matrix for translation in the x direction
    R, T = compute_R_T(E)

    #get K
    K = compute_K(im1)

    #get projection from K, R, T
    P = compute_P(K, R, T)

    #quit()
    Tx = T[2]

    #need to reshape vector so that stereorectifyuncalibrated can
    pts1_n = pts1.reshape((pts1.shape[0] * 2, 1))
    pts2_n = pts2.reshape((pts2.shape[0] * 2, 1))

    #compute rectification transform
    retval, rec1, rec2 = cv2.stereoRectifyUncalibrated(pts1_n, pts2_n, F, (len(im1), len(im1[0])))

    if DEBUGGING:
        print 'rec1:\n', rec1
        print 'rec2:\n', rec2

    #apply rectification matrix
    rec_im1 = cv2.warpPerspective(im1, rec1, (len(im1), len(im1[0])))

    rec_im2 = cv2.warpPerspective(im2, rec2, (len(im2), len(im2[0])))

    if UNIT_TESTING:
        cv2.imshow('im1', rec_im1)
        cv2.waitKey(0)

        cv2.imshow('im2', rec_im2)
        cv2.waitKey(0)

        '''
        h = rec_im1.shape[0]
        w = rec_im1.shape[1]
        dispim = np.zeros((h,2*w), dtype=np.uint8)
        dispim[:,0:w] = rec_im1
        dispim[:,w:] = rec_im2
        for jj in range(0,h,15):
            cv2.line(dispim, (0,jj), (2*w,jj), (255,255,255), 1)
        cv2.imshow('Sbs', dispim)
        cv2.waitKey(0)
        '''

    #get disparity map
    stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16, SADWindowSize=15)

    #rec_im1 = cv2.cvtColor(rec_im1, cv2.COLOR_BGR2GRAY)
    #rec_im2 = cv2.cvtColor(rec_im2, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(rec_im1, rec_im2) #/ 16

    disparityF = disparity.astype(float)
    maxv = np.max(disparityF.flatten())
    minv = np.min(disparityF.flatten())
    disparityF = 255.0*(disparityF-minv)/(maxv-minv)
    disparityU = disparityF.astype(np.uint8)

    print 'disparity.dtype:', disparity.dtype

    cv2.imwrite('disparity.jpg', disparityU)

    if UNIT_TESTING:
        cv2.imshow('disparity', disparityU)
        cv2.waitKey(0)

        plt.subplot(122), plt.imshow(disparityF)
        plt.show()


    #get perspective transform (Q)
    cx = len(im1[0]) / 2
    cy = len(im1) / 2
    cxp = cx

    Q = np.asarray([[1, 0, 0, -cx],
                    [0, 1, 0, -cy],
                    [0, 0, 0, f_pix],
                    [0, 0, -1/Tx, (cx-cxp)/Tx]])

    #reproject to 3d
    im_3D = cv2.reprojectImageTo3D(disparityU, Q)

    if show_rt:
        print '\nget_cld() runtime:', time.time() - t, 'secs\n'

    return im_3D

'''
returns euclid_pts, hom_pts
'''
def triangulate(im1, im2): #im1 and im2 must be black and white
    if show_rt:
        t = time.time()

    #compute Fundamental matrix
    F, pts1, pts2 = compute_F(im1, im2)

    #print '\npts1:\n', pts1
    #print '\npts2:\n', pts2

    #compute Essential matrix
    E = compute_E(F, im1, im2)

    #get translation matrix for translation in the x direction
    R, T = compute_R_T(E)

    #get K
    K = compute_K(im1)

    #get projection from K, R, T
    P = compute_P(K, R, T)

    print '\nP.shape:', P.shape, '\n'
    print '\npts1.T.shape:', pts1.T.shape, '\n'
    print '\npts2.T.shape:', pts2.T.shape, '\n'

    #if pts1.T.shape[1] == 0 or pts.T.shape[0] == 0:
     #   return None, None, None # None, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    hom_pts = cv2.triangulatePoints(P, P, pts1.T, pts2.T)

    #print 'hom_pts:\n', hom_pts.T
    euclid_pts = cv2.convertPointsFromHomogeneous(hom_pts.T)

    xfrm = compute_transform(R, T)

    if show_rt:
        print '\ntriangulate() runtime:', time.time() - t, '\nsecs'

    return euclid_pts, hom_pts.T, xfrm

def ply_from_im(im3d):
    if DEBUGGING:
        print '\nBuildling .ply file\n'

    if show_rt:
        t = time.time()

    h = len(im3d)
    w = len(im3d[0])
    im_size = w * h

    le_file = open('my_ply.ply', 'w')

    #go through and find all the points that arent infinity
    inf_ct = 0
    for i in range(len(im3d)):
        for j in range(len(im3d[0])):
            a = str(im3d[i, j, 0])
            b = str(im3d[i, j, 1])
            c = str(im3d[i, j, 2])

            if (a and b and c) != 'inf':
                if (a and b and c) != '-inf':
                    inf_ct += 1

                    #if DEBUGGING:
                    #    print '\ninf_ct:', inf_ct, '\n'

    if DEBUGGING:
        print '\nLength of im3d:', im_size
        print '# of non-infinity pts in im3d:', inf_ct, '\n'


    le_file.write(( 'ply\n' +
                    'format ascii 1.0\n' +
                    'element vertex ' + str(inf_ct) + '\n'
                    'property float x\n' +
                    'property float y\n' +
                    'property float z\n' +
                    'end_header\n'))

    #unique_pt_ct = 0
    for i in range(h):
        for j in range(w):

            x = str(im3d[i][j][0])
            if x == '-inf' or x == 'inf':
                x = str(1000)

            y = str(im3d[i][j][1])
            if y == '-inf' or y == 'inf':
                y = str(1000)

            z = str(im3d[i][j][2])
            if z == '-inf' or z == 'inf':
                z = str(1000)

            #only write those values that arent infinity
            if (x and y and z) != '1000':
                le_file.write(x + ' ' + y + ' ' + z + '\n')
                #++unique_pt_ct

    le_file.close()

    if DEBUGGING:
        print '\n\t.ply file built\n'

    if show_rt:
        print '\nbuild_ply_file() runtime:', time.time() - t, 'secs\n'

def ply_from_list(le_list):
    if DEBUGGING:
        print '\nBuildling .ply file\n'

    if show_rt:
        t = time.time()

    list_len = len(le_list)

    le_file = open('my_sparse_ply.ply', 'w')

    #go through and find all the points that arent infinity
    inf_ct = 0
    for i in range(list_len):
        for j in range(len(le_list[0, 0])):
            a = str(le_list[i, 0, j])
            b = str(le_list[i, 0, j])
            c = str(le_list[i, 0, j])

            if (a and b and c) != 'inf':
                if (a and b and c) != '-inf':
                    inf_ct += 1

                    #if DEBUGGING:
                    #    print '\ninf_ct:', inf_ct, '\n'

    if DEBUGGING:
        print '\nLength of le_list:', list_len
        print '# of non-infinity pts in le_list:', inf_ct, '\n'


    le_file.write(( 'ply\n' +
                    'format ascii 1.0\n' +
                    'element vertex ' + str(inf_ct) + '\n'
                    'property float x\n' +
                    'property float y\n' +
                    'property float z\n' +
                    'end_header\n'))

    for i in range(list_len):
        for j in range(len(le_list[0, 0])):
            a = str(le_list[i, 0, j])
            b = str(le_list[i, 0, j])
            c = str(le_list[i, 0, j])

            if (a and b and c) != 'inf':
                if (a and b and c) != '-inf':
                    le_file.write(a + ' ' + b + ' ' + c + '\n')

    le_file.close()

    if DEBUGGING:
        print '\n\t.ply file built\n'

    if show_rt:
        print '\nbuild_ply_file() runtime:', time.time() - t, 'secs\n'

def countdown(count):
    for i in range(count):
        print '\n\tstarting in', count - i, 'secs\n'
        time.sleep(1)

    return True

def cld_from_frame(cam, method):

    countdown(cd)
    frame1 = get_frame(cam)
    #cv2.imwrite('frame1.jpg', frame1)

    countdown(cd)
    frame2 = get_frame(cam)
    #cv2.imwrite('frame2.jpg', frame2)

    im1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if method == 'triangulate':
        cld = triangulate(im1, im2)

    if method == 'disparity':
        cld = get_cld(im1, im2)

    return cld, im1, im2

def compute_transform(R, T):
    print 'R.shape:', R.shape
    print 'T.shape:', T.shape

    xfrm = np.c_[R, T]

    xfrm = np.r_[xfrm, [[0, 0, 0, 1]]]
    return xfrm


if show_rt:
    rt = time.time()

if CAM_TEST:
    cam = init_camera(c_no, 288, 384)

    countdown(cd)
    frame1 = get_frame(cam)
    cv2.imwrite('frame1.jpg', frame1)

    countdown(cd)
    frame2 = get_frame(cam)
    cv2.imwrite('frame2.jpg', frame2)

    im1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    cld, hom_cld, x = triangulate(im1, im2)

    for i in range(cam_rd_ct):
        #countdown(cd)
        frame1 = get_frame(cam)
        #countdown(cd)
        time.sleep(.1)
        frame2 = get_frame(cam)
        cv2.imwrite('frame'+str(i+2)+'.jpg', frame1)
        cv2.imwrite('frame'+str(i+3)+'.jpg', frame2)

        im1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        n_cld, n_hom_cld, n_x = triangulate(im1, im2)

       # if n_cld == None:
        #    continue

        #print '\nn_hom_cld.shape:', n_hom_cld.shape, '\n'
        x = np.dot(x, n_x)

        t_hom_cld = np.dot(x, n_hom_cld.T)
        #print '\nt_hom_cld.shape:', t_hom_cld.shape, '\n'
        h_size = hom_cld.shape[0]

        hom_cld = np.r_[hom_cld, t_hom_cld.T]
        #print '\nhom_cld.shape:', hom_cld.shape, '\n'

        if hom_cld.shape[0] > h_size:
            print '\nIt\'s growing...\n'
            print '\n\tIteration', i, '\n'

        #pdb.set_trace()
    #print '\nlen(hom_cld):', len(hom_cld), '\n'
    #print '\nhom_cld.dtype:', hom_cld.dtype, '\n'
    cld = cld.astype(float)
    cld = cv2.convertPointsFromHomogeneous(hom_cld.astype(np.float32))
    ply_from_list(cld)

    cld = get_cld(im1, im2)
    ply_from_im(cld)



if MEM_TEST:
    #im1 = cv2.imread('scene_r.bmp', 0)
    #im2 = cv2.imread('scene_l.bmp', 0)

    im1 = cv2.imread('frame1.jpg', 0)
    im2 = cv2.imread('frame2.jpg', 0)
    cld = get_cld(im1, im2)
    ply_from_im(cld)

    cld, hom_cld, x = triangulate(im1, im2)
    ply_from_list(cld)

if CAM_READ:
    countdown(3)
    cam = init_camera(1, 288, 384)    #set up camera
    ref_im = get_frame(cam)    #get reference image
    countdown(3)
    im1 = get_frame(cam)    #get second image
    e_pts, h_pts, xfrm = triangulate(ref_im, im1)    #compute point cloud
    clds = np.array([[[]]])   #make a list of clouds
    xfrms = np.array([[[]]]) #make a list of transformations
    xfrms = np.append(xfrms, xfrm)#save transformation from reference to first
    clds = np.append(clds, h_pts)

    pdb.set_trace()

    for i in range(cam_rd_ct):  #loop
        im_n = get_frame(cam)   #get images n and n+1
        countdown(2)
        im_n1 = get_frame(cam)
        e_pts_n, h_pts_n, xfrm_n = triangulate(im_n, im_n1) #compute transform between them
        xfrms = np.append(xfrms, xfrm_n)    #save transform as transform n
        clds = np.append(clds, h_pts_n) #save point cloud in point cloud list
    #pdb.set_trace()


    for i in range(1, len(clds), 1):#for all point clouds
        ref_xfrm = xfrms[0]
        for j in range(1, i, 1):    #compute transform between image and reference
            ref_xfrm = np.dot(ref_xfrm, xfrms[j]) #=PI(transform) from 0 to n
        t_cld = np.dot(ref_xfrm, clds[i])   #apply transform to point cloud
        print 't_cld:\n', t_cld
        print '\nclds[0]:\n', clds[0]
        clds[0] = np.r_[clds[0], t_cld]   #concatinate to mian pt cld

if show_rt:
    print '\nTotal Runtime:', time.time() - rt