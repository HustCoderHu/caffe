TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    src/blob_by_filter.cpp \
    src/blob_by_kernel.cpp \
    src/mylayer.cpp \
    src/mynet.cpp \
    src/myblob.cpp \
    src/mysolver.cpp \
    test/test_MyNet.cpp \
    src/blob_by_channel.cpp \
    test/display_weights.cpp
INCLUDEPATH += \
    include/    \
    caffe_src/include/    \
    caffe_src/build/src/

DISTFILES += \
    scpfile.sh \
    CMakeLists.txt

HEADERS += \
    include/blob_by_filter.h \
    include/blob_by_kernel.h \
    include/cpu_only.h \
    include/layer_set.h \
    include/myblob_factory.h \
    include/mylayer.h \
    include/mynet.h \
    include/myblob.h \
    include/mysolver.h \
    include/helper_functions.hpp \
    include/blob_by_channel.h
