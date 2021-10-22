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

# How to run the container
Once you have built the image, you will run it while mounting your input and output directories to get predictions from the model.
For example, you have a folder `tnc_dir` on your computer which contains a directory `input_images`, and you'd like the results of the model to go in `tnc_dir/model_output`. You'll be mounting the `tnc_dir` to the container (at the root directory) and using the `input_images` and `model_output` folders as arguments to the script:
```
$ docker run --mount type=bind,source=tnc_dir,target=/tnc_dir tnc /tnc_dir/input_images /tnc_dir/model_output
```