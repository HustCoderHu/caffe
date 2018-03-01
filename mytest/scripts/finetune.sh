# CUR_DIR=$(pwd)

# 1080ti-hzx
MYDIR="/home/hzx/cnn-hzx"
# MODEL=$CUR_DIR/resnet_50_train_val.prototxt
# WEIGHTS="/home/hzx/caffe/models/resnet50/ResNet-50-model.caffemodel"

# 16041-server
# MYDIR="/root/cnn-hzx"

PR_RATE=0.9
LR_POLICY=inv # decayed
LOG=resnet50-$PR_RATE-$LR_POLICY.log

BIN=/home/hzx/cnn-hzx/mytest/build/mycaffe.bin
RESNET=$MYDIR/resnet-50
# OUTPUT=$RESNET/output
WEIGHTS=$RESNET/ResNet-50-model.caffemodel
SOLVER=$RESNET/retrain-$PR_RATE/$LR_POLICY/max2e-4_min2e-5.solver
TRAIN_NET=$RESNET/resnet_50_train.prototxt
TEST_NET=$RESNET/resnet_50_test.prototxt


cd $RESNET
sudo $BIN $WEIGHTS $SOLVER $PR_RATE  2>&1 | tee $LOG
# $CUR_DIR/a.bin $WEIGHTS $MODEL > output.txt 2>&1
# $CUR_DIR/a.bin $WEIGHTS $TEST_NET > output.txt