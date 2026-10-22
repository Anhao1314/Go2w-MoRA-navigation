# Guardrail: verifiable-arbitration core (architecture v0.2).
#
# Builds as part of the main C++ tree.  Only Eigen and nlohmann/json are
# required; the Go2 parameters are loaded from the JSON exported by
# scripts/export_go2_model.py, so no robot constants are compiled in.

include(FetchContent)

find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found; fetching 3.11.3")
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(guardrail_core STATIC
  src/model.cpp
  src/kinematics.cpp
)

target_include_directories(guardrail_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(TARGET Eigen3::Eigen)
  target_link_libraries(guardrail_core PUBLIC Eigen3::Eigen)
else()
  target_link_libraries(guardrail_core PUBLIC eigen)
endif()

if(TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(guardrail_core PUBLIC nlohmann_json::nlohmann_json)
else()
  target_link_libraries(guardrail_core PUBLIC nlohmann_json)
endif()

if(MSVC)
  target_compile_options(guardrail_core PRIVATE /W4)
else()
  target_compile_options(guardrail_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GO2W_BUILD_TESTS AND TARGET Catch2::Catch2WithMain)
  add_executable(guardrail_tests tests/test_kinematics.cpp)
  target_link_libraries(guardrail_tests PRIVATE guardrail_core Catch2::Catch2WithMain)
  target_compile_definitions(guardrail_tests PRIVATE
    GUARDRAIL_MODEL_PATH="${PROJECT_SOURCE_DIR}/../locomotion/guardrail/go2_model.json"
  )
  add_test(NAME guardrail_tests COMMAND guardrail_tests)
endif()
