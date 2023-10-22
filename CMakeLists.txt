cmake_minimum_required(VERSION 3.9)
project(dlib_test)

set(CMAKE_CXX_STANDARD 14)

find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} dlib::dlib)