import cv2
vidcap = cv2.VideoCapture('zebra1.mp4')
count = 0
success,image = vidcap.read()
while success:
  success,image = vidcap.read()
  if count == 375:
    cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
    break
  print('Read a new frame: ', success)
  count += 1