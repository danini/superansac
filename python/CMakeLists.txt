# Find Pybind11
find_package(pybind11 REQUIRED)

# Python extension module
pybind11_add_module(pysuperansac MODULE
  pybind_module.cpp
  pybind_functions.cpp
)

# Link the C++ core library
target_link_libraries(pysuperansac PRIVATE SupeRANSAC)

# Ensure loader finds the sibling core .so at runtime
set_target_properties(pysuperansac PROPERTIES
  BUILD_RPATH   "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN NO
)

# LTO can stay enabled globally; disable here if needed
# set_property(TARGET pysuperansac PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)

# Install module next to the core .so
install(TARGETS pysuperansac
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .
)
