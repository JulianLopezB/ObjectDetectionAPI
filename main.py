"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
#import argparse
import io
import os
import subprocess
from PIL import Image, ImageFont, ImageDraw
import shutil
import cv2

import torch
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename



model = torch.hub.load('yolov5', 'custom',  path='models/yolov5s.pt',  source='local', force_reload=True)

model.eval()


app = Flask(__name__)


uploads_dir = os.path.join(app.instance_path, 'uploads')

def refresh_paths():
    os.makedirs('static/tmp', exist_ok=True)
    clean_path_content('static/tmp')
    os.makedirs('static/tmp/frames', exist_ok=True)
    print(f'Uploading temporary files to {uploads_dir}')
    os.makedirs(uploads_dir, exist_ok=True)

def clean_path(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

def clean_path_content(folder):

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f'Files in {folder} deleted')
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        refresh_paths()
        filename = file.filename
        is_video =  filename.endswith('.mp4') #or filename.endswith('.vid')
        if not is_video:
            output_path = 'static/tmp/image0.jpg'
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            get_img_predictions(img, output_path)
        else:
            frames_path = 'static/tmp/frames/'
            output_path = 'static/tmp/output.mp4'
            get_video_predictions(file, frames_path)            
            render_video(frames_path, output_path)
        return redirect(output_path)
    return render_template("index.html")


def get_img_predictions(img, output_path):
    results = model(img, size=640)

    # for debugging
    # data = results.pandas().xyxy[0].to_json(orient="records")
    # return data

    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        img_base64 = Image.fromarray(img)
        img_base64.save(output_path, format="JPEG")

def generate_frames(cap):
    
    while True:
            
        success,frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            #frame=buffer.tobytes()
            
        yield frame

def get_video_predictions(file, frames_path):

    filename = 'tmp.mp4'
    vid_input = os.path.join(uploads_dir, secure_filename(filename))
    file.save(vid_input)
    cap=cv2.VideoCapture(cv2.CAP_FFMPEG)
    cap.open(str(vid_input))
    
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        exit()
    
    predictions = []
    for i, frame in enumerate(generate_frames(cap)):
        get_img_predictions(frame, frames_path + f'{i}.jpg')
            
    cap.release()
    cv2.destroyAllWindows()
    
    #return predictions

def get_n_frame(x):
    return int(float(x[:-4]))

def render_video(pathOut_Frames, videoOut_file):
    list_frames = [img for img in os.listdir(pathOut_Frames) if img.endswith('.jpg')]
    list_frames = sorted(list_frames, key = get_n_frame)
    width, height, layers =cv2.imread(pathOut_Frames+list_frames[1]).shape
    print(f'Building a video of size {width}x{height}')
    fps = 15
    capSize = (height, width) 
    fourcc = cv2.VideoWriter_fourcc(*'H264') #*'MJPG'
    out = cv2.VideoWriter()
    success = out.open(videoOut_file, fourcc, fps, capSize, True) 
    if success:
        for img in list_frames:
            img_data = cv2.imread(pathOut_Frames+img)
            out.write(img_data)
    else:
        print('Error creating output video')
        
    out.release()
    out = None
    print("Video done")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    #parser.add_argument("--port", default=5000, type=int, help="port number")
    #args = parser.parse_args()

    #app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
