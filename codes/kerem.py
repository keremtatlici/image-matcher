import cv2

def resize_image(image, divided_by):
    
    return cv2.resize(image, (int(image.shape[1] / divided_by), int(image.shape[0] / divided_by)))

def show_images(images, divided_by=None):
    for idx in range(len(images)):
        if divided_by is not None:
            cv2.imshow(str(idx), resize_image(images[idx], divided_by))
        else:
            cv2.imshow(str(idx), images[idx])
    cv2.waitKey()
    cv2.destroyAllWindows()
