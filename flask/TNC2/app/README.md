# Instructions for Flask App
In order to test the flask app on your machine, it is recommend that you use conda to manage your environment rather than pip. Conda provides an easy installation for
tensorflow-gpu with appropriate installation of CUDA and CUDnn files. 

You can download conda by visiting the website [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and following OS specific instructions. 

In addition, you'd also want to ensure that you have Redis installed on your machine. If you are on Windows, stop here and either test on a Linux/MacOS machine or try installing Ubuntu LTS on your machine and install Redis on there. 
You can follow the instructions [here](https://redis.io/download) and ensure the service is running by pinging the redis-server through the redis-cli. 

Now that you have conda and redis installed, you'd want to create 2 environemnts on conda. One would host the web-server while the other would run our image detection model. 

```bash
conda create --name web-server
conda install flask
pip install flask-uploads flask-executor flask-dropzone redis
```
> Note that creating the environment should automatically default to a python 3.x environment. Ensure that is the case, however. 
```bash
conda create --name web-server python=3.x
```

Create another environemnt for tensorflow as so and install all dependencies:
```bash
conda create --name model-server tensorflow-gpu
conda install matplotlib pandas pillow
pip install redis openpyxl piexif exifread gpsphoto xlsxwriter
```

With 2 terminals now activated with the above environments, you can run the servers as such:
```bash
cd flask/TNC2/app
conda activate web-server
export FLASK_APP=run.py
export FLASK_ENV=development
```

```bash
cd flask/TNC2/app/app
conda activate model-server
python model_server_test.py
```

# Workflow 
The web-server hosts the flask app that accepts files of the format .zip or images/* (includes jpg, png, and other popular image formats). The server has been designed to primarily cater
to jpg images however and it is best to use it with jpg inputs. Instructions are available on the website to have you navigate the routes. An email is necessary to submit images for processing.
Once submitted, the images are tied to a unique userID and stored in-memory via redis. 

The model-server on the other hand, is continuosly polling redis and looking for images to process. If images are queued, the model kicks in and processes the images for us.
The processing section includes running the model, making predicitons on the image, constructing bounding boxes if necessary, extracting metadata, tying all of the aforementioned information 
together in an excel sheet and last but not the least, serving a zip file of the images and the excel file. 

Once complete, the model-server removes the images from the redis queue and stores a zip file with the name userID. The web server, if accessed via the download link, is polling the 
project directory for the presence of a zip file with the name userID. If present, it serves the file to the user. If not, there is a countdown timer indicating an ETC.

> Avoid using the requirements.txt. It is very machine specific and isn't suggested to load the environment. 

- [x] Serve web app on ds4cg-1.cs.umass.edu:5000
- [ ] Write service file in systemd/system to initiate servers on boot. Require root access
- [ ] Write a daily cron job to remove week/2 week old files from DOWNLOAD_PATH
- [ ] Reverse proxy with nginx to free port 5000
- [ ] Switch file serving from send_from_directory to nginx response since files are rather large
