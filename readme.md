rpi image processing 

i am following this video: https://www.youtube.com/watch?v=z70ZrSZNi-8&list=PLxiboCAozahRDCXNN__70bwpnxDt2_u9b&index=2

setting up the raspberry pi. update and upgrade.

mkdir ~/yolo
cd ~/yolo
python3 -m venv --system-site-packages venv
source venv/bin/activate

pip install ultralytics ncnn

yolo detect predict model=yolo11n.pt

for converting to ncnn currently it gives an error, so following this:
https://github.com/ultralytics/ultralytics/issues/19091

pip install ultralytics==8.3.70 torch==2.5.0 torchvision==0.20.0 --no-cache-dir

then: yolo export model=your_model.pt format=ncnn

python yolo_detect.py --model=yolo11n_ncnn_model --source=usb0 --resolution=640x480


