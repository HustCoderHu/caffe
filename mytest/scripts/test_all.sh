# CUR_DIR=$(pwd)

# 1080ti-hzx
MYDIR="/home/hzx/cnn-hzx"
# MODEL=$CUR_DIR/resnet_50_train_val.prototxt
# WEIGHTS="/home/hzx/caffe/models/resnet50/ResNet-50-model.caffemodel"

# 16041-server
# MYDIR="/root/cnn-hzx"

# PR_METHOD=kernel_L1
# PR_METHOD=filter_L1
PR_METHOD=channel_L1
LR_POLICY=inv # decayed

BIN=/home/hzx/cnn-hzx/mytest/build/test_MyNet
RESNET=$MYDIR/resnet-50
RANKED=$RESNET/$PR_METHOD
TEST_NET=$RESNET/resnet_50_test.prototxt

for i in $(seq 1 9); do
    PR_RATE=0.$i
    echo $PR_RATE
    WEIGHTS_DIR=$RANKED/retrain-$PR_RATE/$LR_POLICY
    WEIGHTS=$WEIGHTS_DIR/resnet50_${LR_POLICY}_iter_10000.caffemodel
    cd $WEIGHTS_DIR
    # echo $(pwd)
    LOG=finalAccuracy-$PR_RATE-$LR_POLICY.log
    # echo $LOG
    $BIN $WEIGHTS $TEST_NET 2>&1 | tee $LOG
done

for i in $(seq 1 9); do
    PR_RATE=0.$i
    echo $PR_RATE
    WEIGHTS_DIR=$RANKED/retrain-$PR_RATE/$LR_POLICY
    cd $WEIGHTS_DIR
    LOG=finalAccuracy-$PR_RATE-$LR_POLICY.log
    tail -7 $LOG |grep acc
done
exit