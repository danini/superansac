# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find Boost
find_package(Boost REQUIRED COMPONENTS system)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Recursively find all .cc and .cpp files
file(GLOB_RECURSE SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/include/*.cc
  ${PROJECT_SOURCE_DIR}/include/*.cpp
)

# Core shared library
add_library(SupeRANSAC SHARED
  superansac.cpp
  ${SOURCE_FILES}
)

# Give a unique, stable SONAME to avoid stale copies being picked at runtime
set_target_properties(SupeRANSAC PROPERTIES
  OUTPUT_NAME superansac_core
  SOVERSION 1
  VERSION   1.0.0
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN NO
)

# Public includes
target_include_directories(SupeRANSAC PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Link deps
target_link_libraries(SupeRANSAC
  Eigen3::Eigen
  Boost::system
  ${OpenCV_LIBS}
)

# Install next to the Python module
install(TARGETS SupeRANSAC
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .
)
