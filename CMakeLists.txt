cmake_minimum_required(VERSION 3.24)
project(PGA)

set(CMAKE_CXX_STANDARD 17)
set(SOURCE_LIB extra_tools.cpp CubicSlpine.h tester.h tester.cpp)

add_library(extra STATIC ${SOURCE_LIB})	# Создание статической библиотеки с именем extra

find_library(FFTW_LIBRARY
        NAMES fftw3 fftw)
set(FFTW_LIBRARIES "${FFTW_LIBRARY}")
if(UNIX AND NOT WIN32)
    find_library(FFTW_libm_LIBRARY
            NAMES m)
    list(APPEND FFTW_LIBRARIES "${FFTW_libm_LIBRARY}")
endif()

add_executable(PGA main.cpp)
target_link_libraries(extra ${FFTW_LIBRARIES})
target_link_libraries(PGA extra)