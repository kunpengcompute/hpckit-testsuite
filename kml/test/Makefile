# Makefile
IS_NEED_CONV ?= 0
ifeq ($(CC),gcc)
    COMPILE = gcc
else
    COMPILE = bisheng
endif
CXXFLAGS = -I. -I  $(HPCKIT_PATH)/latest/hmpi/$(COMPILE)/release/hmpi/include
CFLAGS   = -I. -I  $(HPCKIT_PATH)/latest/hmpi/$(COMPILE)/release/hmpi/include 
LDFLAGS  = -lgtest -lpthread -L. -lkfft -lkblas -lklapack_full -lm -lksolver -lmpi -lkscalapack_full -lkipl -lkm -lkvml -lkvsl -lksvml -lkspblas -L$(HPCKIT_PATH)/latest/hmpi/$(COMPILE)/release/hmpi/lib/ -L/usr/local/lib64

TARGET   = test_kml
SOURCES_C = kml_fft_test.c kml_blas_test.c kml_lapack_test.c kml_solver_test.c kml_scalapack_test.c kml_ipl_test.c kml_vml_test.c kml_vsl_test.c kml_svml_test.c kml_math_test.c kml_spblas_test.c
ifeq ($(IS_NEED_CONV),1)
 LDFLAGS += -lkconv -lkfftf -lkffth
 CFLAGS += -DIS_NEED_CONV=1
 CXXFLAGS += -DIS_NEED_CONV=1
 SOURCES_C += kml_conv_test.c
endif

SOURCES_CPP = kml_test_main.cpp
OBJECTS_C = $(SOURCES_C:.c=.o)
OBJECTS_CPP = $(SOURCES_CPP:.cpp=.o)
OBJECTS = $(OBJECTS_C) $(OBJECTS_CPP)

# 默认目标
all: $(TARGET)

# 构建可执行文件
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS) $(CXXFLAGS)

# 编译 C 文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 编译 C++ 文件
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理
clean:
	rm -f *.o $(TARGET)

rebuild: clean all

.PHONY: all clean rebuild