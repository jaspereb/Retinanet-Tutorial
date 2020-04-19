# To run the ROS node
If you haven't used ROS before you will need to go through tutorial 1 of http://wiki.ros.org/ROS/Tutorials to create a catkin workspace. 

Copy the `retinanet_inferencer` folder into your catkin workspace and run `catkin_make` after installing the requirements. Then open `run_inferencer.launch` and edit the model_path to point to your .h5 model file which has been converted to inference mode. You can also change the input and output topic paths. 

You can then run `roslaunch retinanet_inferencer run_inferencer.launch` which will bring up the node. You need to feed `Image` type messages to the `/object_detection_input` topic (`rosrun topic_tools relay /your/camera/feed /object_detection_input`) and there will be visual results on `/object_detection_output_image` which you can view with `rqt_image_view`. There are also `Detection2DArray` messages on the `/object_detection_output` topic. 

# Requirements
* Sensor messages (eg sudo apt-get install ros-melodic-sensor-msgs)
* Geometry messages
* Vision messages
* CV Bridge
* OpenCV2
* Keras Retinanet Installed
