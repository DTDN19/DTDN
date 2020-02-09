<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u baseline.py \
--data-dir /media/HDD-1/home/peixian/chenpeixian/Dataset/ \
-s market1501_ \
-t DukeMTMC-reID_ \
=======
CUDA_VISIVBLE_DEVICES=0,1,2,3 \
python -u baseline.py \
--data-dir /home/data/ \
-s DukeMTMC-reID \
-t market1501 \
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
-a resnet50 \
-b 128 \
--height 256 \
--width 128 \
<<<<<<< HEAD
--logs-dir ./logs/m-d_in_2048/ \
--epoch 40 \
--workers=4 \
--lr 0.08    \
--features 2048 \
# --resume ./logs/duke-market-75.2.pth.tar \
# --evaluate \
# --resume ./logs/duke-market-in-77.4.pth.tar \



=======
--logs-dir ./logs/ \
--epoch 30 \
--workers=4 \
--lr 0.08    \
--features 1024 \
--resume ./logs/duke-market.pth.tar \
# --evaluate
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
# DukeMTMC-reID
# VeRi
# VehicleID_V1.0

# -s VeRi \
# -t VehicleID_V1.0 \

# -s DukeMTMC-reID \
# -t market1501 \

<<<<<<< HEAD
# MSMT17_V2
=======




>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
