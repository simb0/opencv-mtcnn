# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/adrebert/LINUX/Developing/opencv-mtcnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/adrebert/LINUX/Developing/opencv-mtcnn/build

# Include any dependencies generated for this target.
include sample/CMakeFiles/opencvmtcnn.dir/depend.make

# Include the progress variables for this target.
include sample/CMakeFiles/opencvmtcnn.dir/progress.make

# Include the compile flags for this target's objects.
include sample/CMakeFiles/opencvmtcnn.dir/flags.make

sample/CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.o: sample/CMakeFiles/opencvmtcnn.dir/flags.make
sample/CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.o: ../sample/src/OpenCvMtcnn.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/adrebert/LINUX/Developing/opencv-mtcnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object sample/CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.o"
	cd /media/adrebert/LINUX/Developing/opencv-mtcnn/build/sample && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.o -c /media/adrebert/LINUX/Developing/opencv-mtcnn/sample/src/OpenCvMtcnn.cc

sample/CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.i"
	cd /media/adrebert/LINUX/Developing/opencv-mtcnn/build/sample && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/adrebert/LINUX/Developing/opencv-mtcnn/sample/src/OpenCvMtcnn.cc > CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.i

sample/CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.s"
	cd /media/adrebert/LINUX/Developing/opencv-mtcnn/build/sample && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/adrebert/LINUX/Developing/opencv-mtcnn/sample/src/OpenCvMtcnn.cc -o CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.s

# Object files for target opencvmtcnn
opencvmtcnn_OBJECTS = \
"CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.o"

# External object files for target opencvmtcnn
opencvmtcnn_EXTERNAL_OBJECTS =

sample/libopencvmtcnn.so: sample/CMakeFiles/opencvmtcnn.dir/src/OpenCvMtcnn.cc.o
sample/libopencvmtcnn.so: sample/CMakeFiles/opencvmtcnn.dir/build.make
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_calib3d.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_core.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_dnn.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_features2d.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_flann.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_highgui.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_imgcodecs.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_imgproc.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_ml.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_objdetect.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_photo.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_stitching.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_video.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_videoio.a
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.a
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libboost_timer.a
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.a
sample/libopencvmtcnn.so: /usr/lib/jvm/java-11-openjdk-amd64/lib/libjawt.so
sample/libopencvmtcnn.so: /usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so
sample/libopencvmtcnn.so: lib/libmtcnn.a
sample/libopencvmtcnn.so: /usr/local/lib/opencv4/3rdparty/liblibprotobuf.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_imgcodecs.a
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libjpeg.so
sample/libopencvmtcnn.so: /usr/local/lib/opencv4/3rdparty/liblibwebp.a
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libpng.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libtiff.so
sample/libopencvmtcnn.so: /usr/local/lib/opencv4/3rdparty/liblibjasper.a
sample/libopencvmtcnn.so: /usr/local/lib/opencv4/3rdparty/libIlmImf.a
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libavcodec.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libavformat.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libavutil.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libswscale.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libavresample.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libgtk-3.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libgdk-3.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libpangocairo-1.0.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libpango-1.0.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libatk-1.0.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libcairo-gobject.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libcairo.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libgdk_pixbuf-2.0.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libgio-2.0.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libgobject-2.0.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libglib-2.0.so
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libgthread-2.0.so
sample/libopencvmtcnn.so: /usr/local/lib/opencv4/3rdparty/libquirc.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_calib3d.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_features2d.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_flann.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_imgproc.a
sample/libopencvmtcnn.so: /usr/local/lib/libopencv_core.a
sample/libopencvmtcnn.so: /usr/lib/x86_64-linux-gnu/libz.so
sample/libopencvmtcnn.so: /usr/local/lib/opencv4/3rdparty/libittnotify.a
sample/libopencvmtcnn.so: sample/CMakeFiles/opencvmtcnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/adrebert/LINUX/Developing/opencv-mtcnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libopencvmtcnn.so"
	cd /media/adrebert/LINUX/Developing/opencv-mtcnn/build/sample && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencvmtcnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sample/CMakeFiles/opencvmtcnn.dir/build: sample/libopencvmtcnn.so

.PHONY : sample/CMakeFiles/opencvmtcnn.dir/build

sample/CMakeFiles/opencvmtcnn.dir/clean:
	cd /media/adrebert/LINUX/Developing/opencv-mtcnn/build/sample && $(CMAKE_COMMAND) -P CMakeFiles/opencvmtcnn.dir/cmake_clean.cmake
.PHONY : sample/CMakeFiles/opencvmtcnn.dir/clean

sample/CMakeFiles/opencvmtcnn.dir/depend:
	cd /media/adrebert/LINUX/Developing/opencv-mtcnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/adrebert/LINUX/Developing/opencv-mtcnn /media/adrebert/LINUX/Developing/opencv-mtcnn/sample /media/adrebert/LINUX/Developing/opencv-mtcnn/build /media/adrebert/LINUX/Developing/opencv-mtcnn/build/sample /media/adrebert/LINUX/Developing/opencv-mtcnn/build/sample/CMakeFiles/opencvmtcnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sample/CMakeFiles/opencvmtcnn.dir/depend
