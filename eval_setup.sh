#!/bin/bash
# Check if container exists and remove it
if [ "$(docker ps -aq -f name=evalplus-container)" ]; then
    docker rm -f evalplus-container
fi

# Build custom image
docker build -t evalplus-custom -f modified_evalplus/Dockerfile modified_evalplus

# Run container
docker run -d --name evalplus-container \
  -v $(pwd)/results:/results \
  evalplus-custom:latest \
  tail -f /dev/null