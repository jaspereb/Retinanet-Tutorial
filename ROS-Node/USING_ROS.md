#How it works


#Input/Output Format
sensor_msgs/Image message formatted as BRG8. (If it's not BRG8 you can change the CV2 bridge encoding in the source code). If your output images are all the wrong colour then your input format is not BRG and you need to fix that for the detector to work properly. If it works well when this is broken, you probably trained it on the wrong format images as well. 



