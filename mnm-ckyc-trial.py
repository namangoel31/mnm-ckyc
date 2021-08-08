#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib import pyplot as plt
def plt_imshow(title, image):
	# convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.title(title)
	plt.grid(False)
	plt.show()


# In[ ]:


import cv2
import numpy as np
 
img = cv2.imread('unnamed.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.float32)/25
gray = cv2.filter2D(gray,-1,kernel)
edges = cv2.Canny(gray,400,600,apertureSize = 5)
plt_imshow("images", edges)


# In[ ]:


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
contours = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


# In[ ]:


count=0
for c in contours:
    rect = cv2.minAreaRect(c)
    ((x,y),(w,h),r)=rect
    w=int(w)
    h=int(h)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, h-1],
                        [0, 0],
                        [w-1, 0],
                        [w-1, h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h))
    plt_imshow("Image", warped)
    #cv2.imwrite(("extracted_{}".format(count)),warped)
    cv2.imwrite(str(w) + str(h) + '_faces.jpg', warped)
    cv2.waitKey(0)
    count+=1


# In[ ]:




