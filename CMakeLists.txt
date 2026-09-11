cmake_minimum_required(VERSION 3.20)
project(digit_mujoco LANGUAGES CXX)
set(DIGIT_CONTROLLER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../digit-alip-controller" CACHE PATH "Digit ALIP controller checkout")
if(NOT EXISTS "${DIGIT_CONTROLLER_SOURCE}/src/controller/Digit_Controller.cpp")
  message(FATAL_ERROR "Set DIGIT_CONTROLLER_SOURCE to a digit-alip-controller checkout")
endif()
# Generated code includes eigen3 while the controller includes eigen340.
# Resolve both to the vendored version without changing the source checkout.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")
file(CREATE_LINK "${DIGIT_CONTROLLER_SOURCE}/include/eigen340"
  "${CMAKE_CURRENT_BINARY_DIR}/include/eigen3" SYMBOLIC)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
file(GLOB dynamics CONFIGURE_DEPENDS "${DIGIT_CONTROLLER_SOURCE}/src/gen/dyn/*.cc")
file(GLOB kinematics CONFIGURE_DEPENDS "${DIGIT_CONTROLLER_SOURCE}/src/gen/kin/*.cc")
pybind11_add_module(_digit_alip NO_EXTRAS native/bindings.cpp
  "${DIGIT_CONTROLLER_SOURCE}/src/controller/Digit_Controller.cpp"
  "${DIGIT_CONTROLLER_SOURCE}/src/controller/utils.cpp" ${dynamics} ${kinematics})
target_include_directories(_digit_alip PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${DIGIT_CONTROLLER_SOURCE}/include" "${DIGIT_CONTROLLER_SOURCE}/include/controller"
  "${DIGIT_CONTROLLER_SOURCE}/agility")
target_compile_features(_digit_alip PRIVATE cxx_std_17)
set_target_properties(_digit_alip PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/python")
