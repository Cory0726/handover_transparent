import cv2

if __name__ == '__main__':
    img = cv2.imread('data/grconv_width_img.png')

    print(img.shape, img.dtype, img.max(), img.min())