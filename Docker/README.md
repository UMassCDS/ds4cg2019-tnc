# Intro
This folder contains everything necessary to run [Microsoft's Megadetector CameraTraps](https://github.com/Microsoft/CameraTraps) models inside a Docker container for the Nature Conservancy project.

# How to build the Docker image
You must have [Docker](https://www.docker.com) installed on your machine. Navigate this `Docker` folder in the command line and run `docker build .`.
When this is finished you should see a message `Successfully built <image-id>`. You will use this image id to run your container.

# How to run the container
Once you have