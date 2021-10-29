# Intro
This folder contains everything necessary to run [Microsoft's Megadetector CameraTraps](https://github.com/Microsoft/CameraTraps) models inside a Docker container for the Nature Conservancy project.

Note that this assumes your host machine has only a CPU. If you have a Nvidia GPU available, you can use it by changing the base image to a Tensorflow GPU build (the first `FROM` line of the Dockerfile) and using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). See the [Tensorflow Docker instructions](https://www.tensorflow.org/install/docker) for more details.

# How to build the Docker image
You must have [Docker](https://www.docker.com) installed on your machine. Navigate this `Docker` folder in the command line and run `docker build -t tnc .` to build the image and with a `tnc` tag.
When this is finished you should see a message:
```
$ docker build -t tnc
Sending build context to Docker daemon  17.92kB
Step 1/12 : FROM tensorflow/tensorflow:1.14.0-py3
 ---> 4cc892a3babd
...
Successfully built <image-id>
Successfully tagged tnc:latest
```

## AWS Authentication

Note that althought the app will build, it won't run by default due to a credential error from AWS. 
Before final deployment, we'll need to figure out a better way to do this, but for now, you'll need to add the following lines
to the dockerfile before the image will run:

'''
ENV AWS_ACCESS_KEY_ID=your-access-key-id
ENV AWS_SECRET_ACCESS_KEY=your-secret-access-key
ENV AWS_REGION=us-east-2
'''

# How to run the container
Simply run it- because the image now downloads from s3 instead of looking for local files, there's no need to define the input and output directories or to bother mounting them. 
```
$ docker run tnc
```