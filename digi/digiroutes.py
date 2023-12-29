from __main__ import app
from flask import render_template, make_response, url_for, send_file, abort, flash, request, redirect, jsonify, Response, send_file
from os import listdir
from os.path import isfile, join
from io import BytesIO
from PIL import Image
import cv2
import os
import random
import json

@app.route('/vintapi/uploadform', methods=['GET'])
def uploadform():
    return render_template('uploadform.html')

@app.route('/vintapi/v1/eighties', methods=['POST'])
def eighties():
    print(request.files)
    if 'photo' not in request.files:
        return {'MESSAGE': 'Missing image'}, 401
    else:
        file = request.files['photo']
    try:
        print(file.mimetype)
        file_data = Image.open(file)

        processed_image_io = BytesIO()
        file_data.save(processed_image_io, format='JPEG')
        processed_image_io.seek(0)

        return send_file(processed_image_io, mimetype=file.mimetype, as_attachment=True, download_name='processed_image.jpg')
        
    except Exception as e:
        print(f"Exception in postlog: {e}")
        return {'MESSAGE': f"Exception in /api/updatescore {e}"}, 401 

@app.route('/vintapi/v1/nineties', methods=['POST'])
def nineties():
    pass