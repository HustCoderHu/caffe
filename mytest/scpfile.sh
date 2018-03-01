# remote_ip="115.156.157.154"
# remote_dir="/root/.ssh/authorized_keys"
# file_list="repo-rsa.pub"

# 旦哥 asus 1080ti
# remote_ip="192.168.110.244"
remote_ip=115.156.157.164
usr=hzx
remote_dir=/home/$usr/cnn-hzx/mytest
# remote_dir="/home/$usr/cnn-hzx/weights_zerod"

# 公版 1080ti
# remote_ip="115.156.157.153"
# usr="hzx"
# remote_dir="/home/hzx/cnn-hzx/weights_zerod"

# lab 16041-server
# remote_ip="115.156.157.154"
# usr="root"
# remote_dir="/root/cnn-hzx/weights_zerod"

# dorm 16043-server
# remote_ip=192.168.243.131
# usr=hzx
# remote_dir=/home/$usr/cnn-hzx/mytest

# home 16042
# remote_ip="192.168.126.138"
# usr="xiaohu"
# remote_dir="/home/xiaohu/cnn-hzx/mytest"


pkg_id="code.tar"
file_list="include/ src/ test/ scripts/ CMakeLists.txt *.cpp"

echo ${file_list}
echo on
tar cfv $pkg_id ${file_list}

CDIR="cd "${remote_dir}

cmd_r=${CDIR}
cmd_r=${cmd_r}"; rm -rf ${file_list}; tar xf ${pkg_id}"

scp ${pkg_id} $usr@${remote_ip}:${remote_dir}
ssh -t -t -p 22 $usr@${remote_ip} ${cmd_r}

# cmd_r=${CDIR}
# cmd_r=${cmd_r}"; make -j2"
# ssh -t -t -p 22 $usr@${remote_ip} ${cmd_r}