import cv2


def show_image(label, image):
    cv2.namedWindow(label, cv2.WINDOW_NORMAL)
    cv2.imshow(label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
