#!/usr/bin/python
# coding: utf-8
from flask import Flask, request, Response
from detector import Detector
from serialization import make_alto, make_json
import json

with open("config.json") as f:
    config = json.load(f)
detector = Detector(config=config)

app = Flask(__name__)


@app.route('/alto', methods=["POST"])
def process_to_alto():
    file_identifier = request.form['file_identifier']
    image = request.files["image_data"]

    boxes, (width, height) = detector.process(image)

    result = make_alto(boxes, page_width=width,
                       page_height=height, file_identifier=file_identifier)
    return Response(result, mimetype='text/xml')


@app.route('/json', methods=["POST"])
def process_to_json():
    file_identifier = request.form['file_identifier']
    image = request.files["image_data"]

    boxes, (width, height) = detector.process(image)
    result = make_json(boxes, page_width=width,
                       page_height=height, file_identifier=file_identifier)
    return Response(result, mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)
