# Makefile
CXXFLAGS = -O3
LDFLAGS  = -lkulitho -lpthread -lm

TARGET   = test_kulitho
SOURCES  = kulitho_test.cpp
OBJECTS  = $(SOURCES:.cpp=.o)

# ========== 主目标 ==========
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

# ========== 编译规则 ==========
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ========== 清理 ==========
.PHONY: clean run
clean:
	rm -f *.o $(TARGET)