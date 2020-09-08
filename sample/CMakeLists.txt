#configure variables
set (SAMPLE_NAME "app")

#configure directories
set (SAMPLE_PATH "${PROJECT_SOURCE_DIR}/sample")
set (SAMPLE_SRC_PATH  "${SAMPLE_PATH}/src" )
set (SAMPLE_INCLUDE_PATH "${SAMPLE_PATH}/include")

find_package(OpenCV REQUIRED)
find_package(JNI REQUIRED)

include_directories(${JNI_INCLUDE_DIRS})

set(Boost_USE_STATIC_LIBS   ON)
FIND_PACKAGE(Boost 1.58 COMPONENTS filesystem timer REQUIRED chrono)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")

#set includes
include_directories (
    ${SAMPLE_INCLUDE_PATH}
    ${MTCNN_LIB_INCLUDE_PATH}
)

#set sources
file (GLOB SAMPLE_SOURCE_FILES "${SAMPLE_SRC_PATH}/*.cc")
file (GLOB SAMPLE_HEADER_FILES "${SAMPLE_INCLUDE_PATH}/*.h")

#set target executable
ADD_LIBRARY (${SAMPLE_NAME} SHARED ${SAMPLE_SOURCE_FILES} ${SAMPLE_HEADER_FILES})
#add_executable (${SAMPLE_NAME} ${SAMPLE_SOURCE_FILES} ${SAMPLE_HEADER_FILES})

if (WIN32)
set(LINK_OPTIONS shlwapi.lib)
endif()

if ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
set(LINK_OPTIONS -pthread -ldl -static-libgcc -static-libstdc++)
endif()

if (Darwin)
set(LINK_OPTIONS -pthread)
endif()

#SET_TARGET_PROPERTIES(
#    ${SAMPLE_NAME} PROPERTIES
#    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
#    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}"
#    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}"
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
#    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}"
#    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}")


#add the library
target_link_libraries (
    ${SAMPLE_NAME}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${JNI_LIBRARIES}
    ${MTCNN_LIB_NAME}
    ${LINK_OPTIONS})

#test
enable_testing ()
