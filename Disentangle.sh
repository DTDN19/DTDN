CUDA_VISIVBLE_DEVICES=0,1,2,3 \
python -u baseline.py \
--data-dir /home/data/ \
-s DukeMTMC-reID_ \
-t market1501_ \
-a resnet50 \
-b 128 \
--height 256 \
--width 128 \
--logs-dir ./logs/ \
--epoch 40 \
--workers=4 \
--lr 0.08    \
--features 1024 \
# --resume ./logs/duke-market.pth.tar \
# --evaluate
# DukeMTMC-reID
# VeRi
# VehicleID_V1.0

# -s VeRi \
# -t VehicleID_V1.0 \

# -s DukeMTMC-reID \
# -t market1501 \
