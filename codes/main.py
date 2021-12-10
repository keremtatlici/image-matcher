from os import listdir
from os.path import isfile, join
import cv2
from matplotlib import pyplot as plt
import kerem
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-wf", "--wf_image", required=True, help="wide field image path")
ap.add_argument("-mf", "--mf_image", help="medium field image path.")
ap.add_argument("-mfs", "--mf_images",help="İf you have multiple mf image for single wf image, you can give a directory path that only your mf images inside of it" )
ap.add_argument('-wp', '--write_path', help='write path for the result image if wanted.')
ap.add_argument('-s', '--show', action='store_true', help='just pass this argument empty for print screen the result')
args = vars(ap.parse_args())




try:
    wf_image = cv2.imread(args["wf_image"])
except:
    print(f'Cant read the wf image that has been given.Are you sure thats an image? :{args["wf_image"]}')

mf_images= list()
if args['mf_image']:
    mf_images.append(cv2.imread(args['mf_image']))

elif args['mf_images']:
    file_names = [f for f in listdir(args["mf_images"]) if isfile(join(args["mf_images"], f))]
    [mf_images.append(cv2.imread(args['mf_images']+file_name)) for file_name in file_names]

else:
    raise AssertionError("You must either give the --mf_image parameter the image path or the --mf_images parameter the directory path")

sift4mf = cv2.SIFT_create( contrastThreshold=0.016)
sift4wf = cv2.SIFT_create( contrastThreshold=0.002)

kp_wf, des_wf = sift4wf.detectAndCompute(wf_image, None)
print("Number of feature has been created at wf : ", len(kp_wf))

for mf_image in mf_images:

    kp_mf, des_mf = sift4mf.detectAndCompute(mf_image, None)
    print("Number of feature has been created at mf : ", len(kp_mf))

    good = kerem.flann_matcher(des_mf, des_wf)
    print("Number of good matches : ",len(good))

    src_pts = np.float64([ kp_mf[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float64([ kp_wf[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    wf_image = kerem.panorama(wf_image, mf_image, M, (wf_image.shape[1], wf_image.shape[0]))

if args["write_path"]:
    cv2.imwrite(args["write_path"] ,wf_image)
if args["show"]:
    kerem.show_images([wf_image], 4)



