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

BIN=/home/hzx/cnn-hzx/mytest/build/main
RESNET=$MYDIR/resnet-50
RANKED=$RESNET/$PR_METHOD
WEIGHTS=$RESNET/ResNet-50-model.caffemodel
SOLVER=$RESNET/max2e-4_min2e-5.solver

for i in $(seq 1 9 | tac); do
    PR_RATE=0.$i
    echo $PR_RATE
    WEIGHTS_DIR=$RANKED/retrain-$PR_RATE/$LR_POLICY
    echo $WEIGHTS_DIR
    mkdir -p $WEIGHTS_DIR
    
    cd $WEIGHTS_DIR
    LOG=resnet50-$PR_RATE-$LR_POLICY.log
    echo $LOG
    # echo $(pwd)
    $BIN $WEIGHTS $SOLVER $PR_RATE 2>&1 | tee $LOG
done
