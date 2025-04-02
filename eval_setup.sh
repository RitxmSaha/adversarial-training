#!/bin/bash
docker run -d --name evalplus-container \
  -v $(pwd)/evalplus_results:/app \
  -v $(pwd)/host_code:/host_code \
  evalplus-custom:latest \
  tail -f /dev/null