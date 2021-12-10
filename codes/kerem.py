import cv2
import numpy as np

def downsize_image(image, divided_by):
    
    return cv2.resize(image, (int(image.shape[1] / divided_by), int(image.shape[0] / divided_by)))

def upsize_image(image, extended_by):
    
    return cv2.resize(image, (int(image.shape[1] * extended_by), int(image.shape[0] * extended_by)))

def show_images(images, divided_by=None):
    for idx in range(len(images)):
        if divided_by is not None:
            cv2.imshow(str(idx), downsize_image(images[idx], divided_by))
        else:
            cv2.imshow(str(idx), images[idx])
    cv2.waitKey()
    cv2.destroyAllWindows()

def flann_matcher(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 15)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return good

def bf_matcher(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck= False)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)

    return matches[:10]

def panorama(img1, img2, H, size):
   I = np.linalg.inv(H)
   img = np.zeros(size,dtype=np.uint8)
   img.fill(255)
   img = cv2.warpPerspective(img1, np.identity(3), size, img, borderMode=cv2.BORDER_TRANSPARENT)
   img = cv2.warpPerspective(img2, I, size, img, flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
   return img
#good = flann_matcher()
#good = bf_matcher()