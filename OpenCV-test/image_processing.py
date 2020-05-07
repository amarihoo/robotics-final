import cv2
#import matplotlib.pyplot as plt
import numpy as np

#read in captured table image
im = cv2.imread('./table-view.png')

#convert to HSV for masks
im_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
light_red = (0, 70, 50)
dark_red = (10, 255, 255)

light_blue = (110, 50, 50)
dark_blue = (130,255, 255)

mask1 = cv2.inRange(im_hsv, light_red, dark_red)
mask2 = cv2.inRange(im_hsv, light_blue, dark_blue)

#save mask result
mask3 = mask1 | mask2
result = cv2.bitwise_and(im, im, mask=mask3)
cv2.imwrite('result-mask.png',result)

#conver to grayscale for object detection
result2 = cv2.cvtColor(result, cv2.COLOR_HLS2BGR)
im_gray = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)
#blurs lines for continuous contours
tmp = cv2.medianBlur(im_gray, 5)

ret,thresh = cv2.threshold(tmp,1,255,0)
#plt.imshow(thresh, cmap="gray")
#plt.show()

# Find contours, draw on image and save
contours, hierarchy = cv2.findContours(tmp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0,255,0), 3)
cv2.imwrite('result-contours.png',im)

#w = list()
#fv = list()

#for block in image
for cnt in contours:
   
    object = cv2.minAreaRect(cnt)
    #calculate angle
    angle = object[-1]
    
    if angle < -45:
       angle = -(90 + angle)
    else:
       angle = -angle
   
    print('Object angle: {}'.format(angle))
    print('(x,y) (w,h): {}, {}'.format(object[0], object[1]))
    #w.append(object[0])

#attempt at coordinate conversion

# coordinates from simulation    
#print("x=0.45, y=0.155")
#print('x=0.4225, y=-0.1265')
#fv.append([0.45, 0.155])
#fv.append([0.4225, -0.1265])
'''def lin_map_matrix(FV, W):

    return np.transpose([np.linalg.solve(np.transpose(W), fv) for fv in FV])

#transformation = lin_map_matrix(w, fv)

X = np.array([0.45, 0.4225])

x_scale = 0.00625542 #0.00655039
y_scale = 0.00043971 
transformation = np.array([x_scale, 0], [0, y_scale])

print(X.dot(transformation)) '''      