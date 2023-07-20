import os
import cv2
import time
import argparse
from defisheye import Defisheye


import cv2
import os
import time


vkwargs = {"fov": 180,
            "pfov": 120,
            "xcenter": None,
            "ycenter": None,
            "radius": None,
            "angle": 0,
            "dtype": "linear",
            "format": "fullframe"
            }


# force_mag = 'force_40'
# video_folder = f'apple_videos/{force_mag}_apple/'
# video_folder = 'test_712/video_input/videostest_0' # center_small_grasps_x5 top_grasps_x5 bottom_grasps_x5


# Parent folder containing the video folders
video_folder = 'test_712/video_frames'

# Get a list of all the JPG image files in the folder
jpg_files = [file for file in os.listdir(video_folder) if file.endswith('.jpg')]

start_time = time.time()


# # Get a list of all the JPG image files in the folder
# jpg_files = [file for file in os.listdir(video_folder) if file.endswith('.jpg')]

# start_time = time.time()
# print("Start unwarping")

# Loop through each JPG image file
for jpg_file in jpg_files:
    # Construct the full path to the JPG image file
    jpg_path = os.path.join(video_folder, jpg_file)

    # Read the JPG image
    og = cv2.imread(jpg_path)

    # Apply Defisheye transformation
    # start = time.time()
    obj = Defisheye(jpg_path, **vkwargs)
    x, y, i, j = obj.calculate_conversions()
    # end_class = time.time()
    unwarped = obj.unwarp(og)
    # end_warp = time.time()

    # print("\nInstantiate Class Time: ", end_class - start)
    # print("\nWarp Time: ", end_warp - end_class)
    
    # cv2.imshow("original", og)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()

    # cv2.imshow("undistorted", unwarped)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()

    # Save the unwrapped image
    unwrapped_file = os.path.splitext(jpg_file)[0] + '_unwrap.jpg'
    unwrapped_path = os.path.join(video_folder, unwrapped_file)
    cv2.imwrite(unwrapped_path, unwarped)

end_time = time.time()
print('Unwarping complete.')
print(f'Unwarping time: {end_time - start_time}s')


