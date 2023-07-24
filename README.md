# Tactile_Sensor

Use raw_data_process.py to get the raw data from the raw video gcode and weights file.
Use get_input_frames_new.py and unpack_force_data.py to get the frames of the video input and pressure input. Use unwrap_fisheye.py to unwrap the video input frames for training.
Use train_new_data.py to train the U net model(which is defined in Unet_model.py) and save the model as .pth file.
Use predict_window.py size to predict and validate.
