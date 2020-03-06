import cv2
import numpy as np
from skimage.morphology import erosion
import math

#--------------------------------------------------
def filter_size(image,size=10,connectivity_val=2):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=connectivity_val)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= size:
            img[output == i + 1] = 255
    return img

#--------------------------------------------------
def crop_face(image_edges,image):
    img=image_edges
    height = img.shape[0]
    width = img.shape[1]
    #crop from left
    sum = 0
    left = 0
    for i in range(0,img.shape[1]):
        for j in range (0,img.shape[0]):
            sum=sum+img[j][i]
            avg=sum/height
        if avg>50:
            left = i
            break
    image = image[0: image.shape[0],left:image.shape[1]]
    img=img[0: img.shape[0],left:img.shape[1]]



    # crop from right

    sum = 0
    right = 0
    for i in range(img.shape[1]-1,0,-1):
        for j in range(img.shape[0]-1,0,-1):
            sum=sum+img[j][i]
            avg=sum/height
        if avg>50:
            right=i
            break
    image=image[0: image.shape[0],0:right]
    img = img[0: img.shape[0], 0:right]





    #crop from top

    sum = 0
    top = 0
    for i in range(0,img.shape[0]):
        for j in range (0,img.shape[1]):
            sum=sum+img[i][j]
            avg=sum/height
        if avg>50:
            top = i
            break
    image = image[top: image.shape[0],0:image.shape[1]]
    img = img[top: img.shape[0],0:img.shape[1]]



    # crop from bottom

    sum = 0
    bottom = 0
    for i in range(img.shape[0]-1,0,-1):
        for j in range(img.shape[1]-1,0,-1):
            sum=sum+img[i][j]
            avg=sum/height
        if avg>50:
            bottom=i
            break
    crop_img=image[0: bottom,0:image.shape[1]]

    return crop_img


#--------------------------------------------------
def crop_img(image, new_height=300):
    height = image.shape[0]
    width = image.shape[1]

    aspect_ratio = width/height
    new_width = int(new_height*aspect_ratio)

    if new_height>height and new_width>width:
        #Enlarge
        method=cv2.INTER_LINEAR
    else :
        #En
        method=cv2.INTER_CUBIC
    image = cv2.resize(image, (new_width,new_height) ,interpolation = method)
    return image


def crop_mouth(image,thresh):
    thresh = ~(thresh)
    w = image.shape[1]
    height = image.shape[0]
    middle = int(w / 2)

    center=getCenter(thresh)
    for i in range(0,center):
        thresh[i][center]=122
    point1 = horse_up(thresh, 15, center, height)
    point2 = horse_up(thresh, 15, center + 20, height)
    point = min(point1, point2)


    for i in range(height-1,point,-1):
        image[i][center]=122
    cv2.imwrite("images/SingleCode/crop.png",image)
    image = image[0:point+15, 0:w]
    return image

def detect_axis_x(image,y=91):
    width=image.shape[1]
    height=image.shape[0]

    middle1=int(width/2)

    center=getCenter(image)


    point_l=horse_left(image,10,y,start=10)
    point_r = horse_right(image,10,y,start=width-10)

    x1 = int((center + point_l) / 2.15)
    x2 = int((point_r + center) / 1.9)
    x1=x1-6
    x2=x2+6

    return image,x1,x2



def detect_axis_y(thresh,image):
    image=cv2.threshold(image, 22, 255, cv2.THRESH_BINARY)[1]
    width = thresh.shape[1]
    height = thresh.shape[0]

    middle1 = int(width / 2)
    middle_p = getCenter(image)

    center_p =horse_down(image,15,middle_p,start=20)


    for i in range(0, center_p):
        image[i][middle_p] = 255

    center_p=center_p+10

    for i in range(0, width):
        thresh[center_p][i] = 255

    bone_width,dist1,dist2=getBoneWidth(thresh)
    point_n=dist1+int(bone_width/4.24)
    point_m=middle_p+int(bone_width/4.24)
    #print("Bone Width : " + str(bone_width) + " point_n: " + str(point_n) + " point_m: " + str(point_m))

    s1 = horse_down(image, 3, point_n, start=center_p)
    s2 = horse_down(image, 3, point_m, start=center_p)

    s1 = int(((height-s1)/2)+s1)
    s2 = int(((height - s2) / 2) + s2)
    d=int((s1+s2)/2)
    for i in range(center_p,d):
        thresh[i][point_n+1] = 255
        image[i][point_n+1]=255
    for i in range(center_p, d):
        thresh[i][point_m+1] = 255
        image[i][point_m + 1] = 255
    d=d-6
    return thresh,image,d

#------------------------------------------------------------------------------------------------------
import random
def getCenter(thresh):
    width = thresh.shape[1]
    height = thresh.shape[0]
    #thresh=~thresh
    middle1 = int(width / 2)
    # Line from the top to center of image
    center_p = horse_down(thresh,10,middle1,40)
    center_p=center_p-30


    # Distance from left
    point_l = horse_left(thresh,15,center_p,start=10)

    # Distance from Right
    point_r = horse_right(thresh,15,center_p,start=width-1)
    dist1 = point_l
    dist2 = width - point_r

    medium = int((dist1 - dist2) / 2)
    middle_p = middle1 + medium
    for i in range(0,point_l):
        thresh[center_p][i]=122
    for i in range(width-1, point_r,-1):
        thresh[center_p][i] = 122

    #cv2.imwrite("images/results2/abs" + str(random.randint(0, 9)) + ".png", thresh)
    return middle_p


def getBoneWidth(thresh):
    width = thresh.shape[1]
    # thresh=~thresh
    middle1 = int(width / 2)
    # Line from the top to center of image
    center_p = horse_down(thresh, 10, middle1, 40)
    center_p = center_p - 30

    # Distance from left
    point_l = horse_left(thresh, 15, center_p, start=10)

    # Distance from Right
    point_r = horse_right(thresh, 10, center_p, start=width - 10)

    dist1 = point_l
    dist2 = width - point_r
    bone_width = width - (dist1 + dist2)
    return bone_width,dist1,dist2

def is_contour_bad(cnt,width,height):
    # approximate the contour
    #peri = cv2.arcLength(c, True)
    #approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    centroid_x=int(width/2)
    centroid_y=int(height/2)

    M = cv2.moments(cnt)
    cx = int(M['m10'] / (M['m00']+1))
    cy = int(M['m01'] / (M['m00']+1))
    area = cv2.contourArea(cnt)
    distance_centroid_x=math.fabs(centroid_x-cx)
    distance_centroid_y = math.fabs(centroid_y-cy)
    #print("centroid_x: "+str(distance_centroid_x)+"  centroid_y: "+str(distance_centroid_y))
    #print("Area: " + str(area) + " x: " + str(cx) + " y: " + str(cy))
    if(area<7500 and cx>40 and cy>40 and distance_centroid_x>200 and distance_centroid_y>20 and cv2.arcLength(cnt,True)):
       # print("Area: " + str(area) + " x: " + str(cx) + " y: " + str(cy))
       # print("centroid_x: " + str(distance_centroid_x) + "  centroid_y: " + str(distance_centroid_y))
        return True
    # the contour is 'bad' if it is not a rectangle
    return False
# -------------------------------------------------------------------------

def is_contour_bad_noise(cnt,width,height):
    centroid_x=int(width/2)
    centroid_y=int(height/2)

    M = cv2.moments(cnt)
    cx = int(M['m10'] / (M['m00']+1))
    cy = int(M['m01'] / (M['m00']+1))
    area = cv2.contourArea(cnt)
    distance_centroid_x=math.fabs(centroid_x-cx)
    distance_centroid_y = math.fabs(centroid_y-cy)

    if (area <250 and distance_centroid_x > 40 and distance_centroid_y >  30 ):
        return True
    return False
#------------------------------------------------
def is_noise(cnt, width, height):
    centroid_x = int(width / 2)
    centroid_y = int(height / 2)



    if (area < 250):
        return True
    return False


# ------------------------------------------------
def is_edges_bad_noise(cnt, width, height):
    centroid_x = int(width / 2)
    centroid_y = int(height / 2)

    M = cv2.moments(cnt)
    cx = int(M['m10'] / (M['m00'] + 1))
    cy = int(M['m01'] / (M['m00'] + 1))
    area = cv2.contourArea(cnt)
    distance_centroid_x = math.fabs(centroid_x - cx)
    distance_centroid_y = math.fabs(centroid_y - cy)
    if (area < 290 and distance_centroid_x > 40 and distance_centroid_y >  30):
        return True

    return False
#---------------------------------------------------------------------
def checkBrightness(image):
    set1=[]
    set2=[]
    width = image.shape[0]
    height = image.shape[1]
    for i in range(0,width):
        for j in range(0, height):
            value = image[i][j]
            if value >= 50 and value < 100: set1.append(value)
            elif value>=100 and value < 150:set2.append(value)
    #print(str(len(set2)-len(set1))+" Def: "+str(width*height))

    return int(len(set2)-len(set1)) < 50000

def decreaseBrightness(image):
    width = image.shape[0]
    height = image.shape[1]
    for i in range(0, width):
        for j in range(0, height):
            value = image[i][j]
            if value >= 100 and value < 150: image[i][j] = image[i][j] - 70
    return image
def correctColors(image):
    image = np.uint8(image)
    max=0
    for i in range(0,image.shape[0]):
        for j  in range(0,image.shape[1]):
            if image[i][j]>max:
                max=image[i][j]
    max=int(max/255)

    for i in range(0,image.shape[0]):
        for j  in range(0,image.shape[1]):
            image[i][j]=int(image[i][j]/max)
    image=np.uint8(image)
    return image

#---------------------------- Y Up to down-----------------------------------
def horse_down(image,noise_thickness,x,start=1):
    width = image.shape[1]
    hight = image.shape[0]

    half_width = int( width / 2)
    pixel = 0
    for i in range(start, hight-noise_thickness-1):

        first_order = int(image[i][x]) - int(image[i - 1][x])

        if first_order == -255:
            first_order = int(image[i+2][x]) - int(image[i - 1][x])
            pixel = i
            if first_order == -255:
                first_order = int(image[i+noise_thickness][x]) - int(image[i - 1][x])
                pixel = i

                if first_order == -255:
                    pixel = i;
                    break

    return pixel


# ----------------------------From the Left-----------------------------------

def horse_left(image, noise_thickness,y,start=1):
    width = image.shape[1]

    point = 0
    for i in range(start, width-1):
        first_order = int(image[y][i]) - int(image[y][i - 1])
        if first_order == -255:
            first_order = int(image[y][i + 1]) - int(image[y][i - 1])
            if first_order == -255:
                first_order = int(image[y][i + noise_thickness]) - int(image[y][i - 1])
                point = i
                if first_order == -255:
                    point = i;
                    break
    return point
#----------------------------From the bottom-----------------------------------

def horse_up(image,noise_thickness,x,start=1):
    width = image.shape[1]
    hight = image.shape[0]

    point = 0
    for i in range(start-noise_thickness,0,-1):
        first_order = int(image[i-1][x]) - int(image[i][x]);
        if first_order == -255:
            first_order = int(image[i-1][x]) - int(image[i + 1][x])
            point = i
            if first_order == -255:
                first_order = int(image[i-1][x]) - int(image[i+noise_thickness][x])
                point = i
                if first_order == -255:
                    point = i;
                    break
    return point


#----------------------------From the Right-----------------------------------
def horse_right(image, noise_thickness,y,start=1):
    point = 0
    for i in range(start-noise_thickness, 0, -1):
        first_order = int(image[y][i + 1])-int(image[y][i])
        if first_order == 255:
            first_order = int(image[y][i + 1])-int(image[y][i-1])
            point = i
            if first_order == 255:
                first_order = int(image[y][i + 1])-int(image[y][i-noise_thickness])
                #print(str(i + noise_thickness)+" M_M  "+str(i))
                point = i
                if first_order == 255:
                    point = i;
                    break
    return point


#---------------------------------------Sobel -------------------------------------------
from scipy import signal
def sobel_filter(img, k_size):
    width=img.shape[1]
    height=img.shape[0]
    image = np.zeros((height, width))


    for i in range(100,height-2):
        for j in range(100,width-100):
            if img[i][j]>=25 and img[i][j]<=150 and (img[i+1][j]<=20 or img[i-1][j]<=20):
                image[i][j]=255

    for i in range(100,height-2):
        for j in range(100,width-100):
            if img[i][j]>=25 and img[i][j]<=150 and (img[i][j+1]<=20 or img[i][j-1]<=20):
                image[i][j]=255


    return image


def remove_edges_noise(edges):
    img, contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(edges.shape[:2], dtype="uint8") * 255
    width = edges.shape[1]
    height = edges.shape[0]
    for c in contours:
        if is_edges_bad_noise(c, width, height):
            cv2.drawContours(mask, [c], -1, 0, -1)
    output = cv2.bitwise_and(edges, edges, mask=mask)
    return output


def getFeatures(image):
    height = image.shape[0]
    width = image.shape[1]
    w = 0
    b = 0
    g = 0
    for i in range(0, height):
        for j in range(0, width - 1):
            if (image[i][j] > 180):
                w = w + 1

            elif (image[i][j] < 25):
                b = b + 1

    area = image.shape[0] * image.shape[1]
    g = area - (w + b)

    (means, stds) = cv2.meanStdDev(image)
    # stats = np.concatenate([means, stds]).flatten()

    ent = entropy(image).item()

    mean = np.mean(image).item()
    median = np.median(image).item()
    variance = np.var(image).item()
    SD = np.std(image, axis=None, dtype=None, out=None, ddof=0, keepdims=False).item()
    Skenss = ((mean - median) / SD)

    g = np.full((200, 1), g)
    b = np.full((200, 1), b)
    mean = np.full((200, 1), mean)
    median = np.full((200, 1), median)
    SD = np.full((200, 1), SD)
    variance = np.full((200, 1), variance)
    Skenss = np.full((200, 1), Skenss)
    ent = np.full((200, 1), ent)
    return g, b, mean, median, variance, Skenss, ent


def entropy(signal):
    signal = signal.ravel()
    lensig = signal.size
    symset = list(set(signal))
    propab = [np.size(signal[signal == i]) / (1.0 * lensig) for i in symset]
    ent = np.sum([p * np.log2(1.0 / p) for p in propab])
    return ent
