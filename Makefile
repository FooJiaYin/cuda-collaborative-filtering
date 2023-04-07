CXX := g++
CXXFLAGS := -std=c++11 -O3
NVFLAGS := $(CXXFLAGS)
TARGET := eval
SEQUENTIAL := cf


.PHONY: all
all: $(TARGET)

.PHONY: eval
eval: final.cu
	nvcc $(NVFLAGS) -o eval final.cu -arch=sm_60 -Dpredict=false -Deval=true

.PHONY: predict
predict: final.cu
	nvcc $(NVFLAGS) -o predict final.cu -arch=sm_60 -Dpredict=true -Deval=false

.PHONY: debug
debug: final.cu
	nvcc $(NVFLAGS) -o debug final.cu -arch=sm_60 -Dpredict=true -Deval=true

.PHONY: v2
v2: final_v2.cu
	nvcc $(NVFLAGS) -o v2 final_v2.cu -arch=sm_60 -Dpredict=false -Deval=true

.PHONY: v1
v1: final_v1.cu
	nvcc $(NVFLAGS) -o v1 final_v1.cu -arch=sm_60 -Dpredict=false -Deval=true

.PHONY: seq
seq: final_seq.cc
	$(CXX) $(CXXFLAGS) -o seq final_seq.cc -Dpredict=false -Deval=true

.PHONY: clean
clean:
	rm -f $(TARGET) $(SEQUENTIAL)


