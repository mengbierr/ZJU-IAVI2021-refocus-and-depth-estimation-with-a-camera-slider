cmake_minimum_required(VERSION 3.18)

project(epi-depth)

find_package(OpenCV CONFIG REQUIRED)
find_package(OpenMP)

add_executable(${PROJECT_NAME}
    main.cpp
    guidedfilter.cpp
    guidedfilter.h
)

target_compile_features(${PROJECT_NAME}
    PRIVATE cxx_std_20
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE opencv_core opencv_calib3d opencv_imgproc opencv_videoio opencv_highgui OpenMP::OpenMP_C
)