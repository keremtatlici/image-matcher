# Image Matcher

Automatically match and overlap zoom (medium field / mf / zoom) images to a zoomed out (widefield / wf / wide) image taken at the same time.

```
usage: main.py [-h] -wf WF_IMAGE [-mf MF_IMAGE] [-mfs MF_IMAGES] [-wp WRITE_PATH] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -wf WF_IMAGE, --wf_image WF_IMAGE
                        wide field image path
  -mf MF_IMAGE, --mf_image MF_IMAGE
                        medium field image path.
  -mfs MF_IMAGES, --mf_images MF_IMAGES
                        İf you have multiple mf image for single wf image, you can give a directory path that only your mf images inside of it
  -wp WRITE_PATH, --write_path WRITE_PATH
                        write path for the result image if wanted.
  -s, --show            just pass this argument empty for print screen the result
```

if you have one mf sample:
`python3 codes/main.py -wf datasets/wf/wf.JPG -mf datasets/mf/mf00.JPG -s`

if you have multiple mf sample:
`python3 codes/main.py -wf datasets/wf/wf.JPG -mfs datasets/mf/ -s`

## Before : 
![wf-min](https://user-images.githubusercontent.com/33085629/145640258-dd0bc1f1-8074-48e0-85c7-07fa199a6dbe.JPG)
## After:
![wf_ALL](https://user-images.githubusercontent.com/33085629/145640342-dd817264-838c-4c90-94d8-e38fd3ca2cdf.JPG)
