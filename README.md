Object Detection API and webapp that uses [yolov5](https://github.com/ultralytics/yolov5) pretrained model.

YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development. 

![](static/example.jpg)

# Steps
## Deploy Locally
```bash
$ pip install virtualenv
$ virtualenv objectdetectionapi
$ source objectdetectionapi/bin/activate
$ git clone https://github.com/JulianLopezB/ObjectDetectionAPI.git
$ cd ObjectDetectionAPI
$ git clone https://github.com/ultralytics/yolov5.git
$ pip install -r requirements.txt
$ python main.py
```

then go to your local server

## Deploy on Cloud
### Cloud Run (see [here](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service#deploy))
```bash
$ gcloud builds submit
```

# To Do:
- [x] object detection in images (jpg, jpeg, png)
- [x] object detection in videos (mp4, avi, mkv)
- [ ] add progress bar to webapp
- [ ] add Dockerfile
- [ ] create dropdown in webapp that allows choosing different models
- [ ] deploy to cloud