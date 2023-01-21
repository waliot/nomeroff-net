import os
import sys
import traceback

import cv2
import numpy as np
import ujson
from flask import Flask
from flask import request

from nomeroff_net import __version__
from nomeroff_net import pipeline
from nomeroff_net.pipes.number_plate_classificators.options_detector import CLASS_REGION_ALL
from nomeroff_net.tools import unzip
from waliot._paths import nomeroff_net_dir

print("[INFO], nomeroff net root dir", nomeroff_net_dir)
number_plate_detection_and_reading = pipeline("multiline_number_plate_detection_and_reading",
                                              prisets={
                                                  "ru": {
                                                      "for_regions": CLASS_REGION_ALL,
                                                      "model_path": "latest"
                                                  },
                                              },
                                              default_label="ru",
                                              image_loader=None)

app = Flask(__name__)


@app.route('/version', methods=['GET'])
def version():
    return __version__


@app.route('/detect', methods=['GET'])
def detect():
    data = request.get_json()
    img_path = data['path']
    try:
        result = number_plate_detection_and_reading([img_path])
        (images, images_bboxs,
         images_points, images_zones, region_ids,
         region_names, count_lines,
         confidences, texts) = unzip(result)

        return ujson.dumps(dict(res=texts, img_path=img_path))
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        return ujson.dumps(dict(error=str(e), img_path=img_path))


@app.route('/magic', methods=['POST'])
def magic():
    file = request.files['file'].read()

    try:
        (images, images_bboxs,
         images_points, images_zones, region_ids,
         region_names, count_lines,
         confidences, texts) = fileToImage(file)

        return ujson.dumps(dict(res=texts))
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        return ujson.dumps(dict(error=str(e)))


def fileToImage(file):
    image = cv2.imdecode(np.frombuffer(file, dtype=np.uint8), 1)
    return unzip(number_plate_detection_and_reading([image]))


if __name__ == '__main__':
    app.run(debug=False,
            port=os.environ.get("PORT", 8888),
            host='0.0.0.0',
            threaded=False,
            processes=1)
