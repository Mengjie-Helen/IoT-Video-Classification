# IoT-Video-Classification

The project aims to build a model that can automatically recognize what human activities (one or more activities) a given video contains. 
For simplicity, the focus now is to distinguish 5 different human activities operated on a Fridge, which are no_interaction, open_close_fridge, put_back_item, take_out_item and screen_interaction. Each video has only one activity operated by one human.

Then we used VideoPose3D to extract the 3D pose information into the form of npy files. Then we looked at different keypoints (total 17 keypoints) and used them as features. (VideoPose3D demonstrated that 3D human poses in the video can be effectively estimated with a dilated temporal convolutional model over 2D keypoint sequences. First, we deployed PyTorch dependency and followed the Detectron2 to generate 2D keypoint predictions from videos and used the visualization feature to render each video of the 3D joint predictions.

For each video, there are 3 coordinates and 17 key points. For each video data , the columns are key points in x, y, and z coordinates (e.g. k0_x, k0_y, k0_z, k1_x, k1_y, k1_z, …, k50_x, k50_y, k50_z,), and the rows represent each frame. 
HIP = 0
Here are the keypoints:   R_HIP = 1, R_KNEE = 2, R_FOOT = 3, L_HIP = 4,  L_KNEE = 5,  L_FOOT = 6, SPINE = 7,  THORAX = 8,  NOSE = 9,  HEAD = 10, L_SHOULDER = 11,  L_ELBOW = 12, L_WRIST = 13, R_SHOULDER = 14, R_ELBOW = 15, R_WRIST = 16.
