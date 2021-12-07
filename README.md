# ImageDetector

Simple wrapper around [dhSegment](https://github.com/dhlab-epfl/dhSegment) network trained to recognize images in scanned documents. Processing single document takes 2.5 to 3 seconds. The result can be saved as json or ALTO containing positions of bounding boxes around images.

You can use it to process local files or start a server, see below.

# Installation

Install conda and run:

```
conda env create -f environment.yml
conda activate image_detector
conda install tensorflow=1.13.1
```

If you have a gpu, use `tensorflow-gpu=1.13.1` instead.

Model weights had to be split to multiple parts to be stored on github. To join them together, run:

`python join_model_weights.py`

# Process local files.

See `demo_local.ipynb` for a script that processes local files.

# Start server

Run `python app.py` to start flask server at http://127.0.0.1:5000. You can then send scanned document together with file identifier (neccessary for ALTO format) in a POST request to http://127.0.0.1:5000/alto or http://127.0.0.1:5000/json, the response will contain ALTO.xml or json data with positions of images in the document.

To send image with curl:

```
curl \
  -F "file_identifier=42" \
  -F "image_data=@./demo_images/0b0e8830-aaf5-11e6-adc9-5ef3fc9ae867.jpg" \
  http://127.0.0.1:5000/alto

```

To send image with python, see `demo.ipynb`.
