import os
import sys
import traceback

import ujson
from flask import Flask
from flask import request
from flask_wtf.csrf import CSRFProtect

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
                                              image_loader="opencv")

csrf = CSRFProtect()
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


if __name__ == '__main__':
    csrf.init_app(app)
    app.run(debug=False,
            port=os.environ.get("PORT", 8888),
            host='0.0.0.0',
            threaded=False,
            processes=1)
