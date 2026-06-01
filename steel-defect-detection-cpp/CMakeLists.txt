cmake_minimum_required(VERSION 3.16)
project(steel_defect_detection_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Drogon (High-performance C++ Web Server)
find_package(Drogon REQUIRED)

# Find SQLite3
find_package(SQLite3 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
)

# Source files
set(SOURCE_FILES
    src/main.cpp
    src/YoloDetector.cpp
    src/DbManager.cpp
)

# Header files
set(HEADER_FILES
    src/YoloDetector.hpp
    src/DbManager.hpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    Drogon::Drogon
    ${OpenCV_LIBS}
    ${SQLite3_LIBRARIES}
    onnxruntime # Links ONNX Runtime C++ API
)

# Set compile flags
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX /EHsc)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Werror)
endif()
