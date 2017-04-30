# Behaviorial Cloning Project

Overview
---
The main files of this repo are the training.py and the writeup.md. The result of driving the vehicle autonomously is captured in run.mp4. 

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:

* drive.py
* video.py
* training.py
* writeup.md

The simulator can be downloaded from the download-data.sh file.  

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. Note that the model.h5 used in this repo 
is available in the S3 bucket but for AWS cost reasons the file is not available for download. 

Once the model has been created locally, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time 
and send the predicted angle back to the server via a websocket connection.
