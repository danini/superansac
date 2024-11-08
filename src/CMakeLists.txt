# Find Eigen3
find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

# Find Boost
find_package(Boost REQUIRED COMPONENTS system)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Recursively find all .cc and .cpp files
file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/include/*.cc ${PROJECT_SOURCE_DIR}/include/*.cpp)

# Add library target
add_library(SupeRANSAC SHARED
    superansac.cpp
    ${SOURCE_FILES}
)

# Include directories for the library
target_include_directories(SupeRANSAC PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Link Eigen, Boost, and OpenCV
target_link_libraries(SupeRANSAC 
    Eigen3::Eigen
    Boost::system
    ${OpenCV_LIBS}
)
