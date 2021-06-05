import os
import base64
from io import BytesIO
from evaluate import enhance
from PIL import Image
from flask import Flask, json, request, jsonify, send_file, render_template
from func_timeout import func_timeout, FunctionTimedOut


app = Flask(__name__)


def image_to_string(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    string_img = base64.b64encode(buffered.getvalue()).decode('ascii')
    return f"data: image/jpeg; base64, {string_img}"


if os.environ.get("PROD") != "true":
    print("Starting in dev")
    from flask_cors import CORS
    CORS(app)

timeout = int(os.environ.get("TIMEOUT", 10))
gpu = int(os.environ.get("GPU", 0))


@app.route("/api/timeout", methods=["GET"])
def get_timeout():
    return jsonify({'timeout': timeout})


@app.route("/api/enhance", methods=["POST"])
def process_image():
    file = request.files['file']
    # Read the image via file.stream
    try:
        lr = Image.open(file.stream)
        sr, depth = func_timeout(timeout, enhance, args=(lr, gpu))[0]

        sr_img = image_to_string(sr)
        depth_str = image_to_string(depth)

        return jsonify({'success': True, "sr": sr_img, 'depth': depth_str})
    except FunctionTimedOut:
        return jsonify({'success': False, 'msg': f"The process took more than {timeout} seconds. Maybe try a smaller picture."})
    except Exception as e:
        print(e)
        return jsonify({'success': False, 'msg': "There was a problem"})


@app.route('/')
@app.route('/demo')
def home():
    return send_file("frontend/dist/index.html")


@app.route('/<path:path>')
def static_file(path):
    return send_file(f"frontend/dist/{path}")


if __name__ == "__main__":
    host = os.environ.get('HOST', "localhost")
    port = int(os.environ.get('PORT', 5000))
    print(f"Server running on {host}:{port}")
    app.run(host, port, debug=(os.environ.get("PROD") != "true"))
