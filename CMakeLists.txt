cmake_minimum_required(VERSION 3.28) # Minimum required for module support
project(evosim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
message (${CMAKE_CXX_COMPILER})

# Set the C++ standard
add_compile_options(-g)
add_compile_options(-Wall)
add_compile_options(-Wno-deprecated-declarations)
add_compile_options(-fdiagnostics-color=always)
add_compile_options(-v)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(spdlog CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

# Add the `src` directory
add_subdirectory(src)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("-- Use flag -fsanitize=address")
    list(APPEND CMAKE_CXX_FLAGS "-fsanitize=address")
endif()

# Exports compile_commands.json to project root for ide support
add_custom_target(
    copy-compile-commands ALL
    ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_CURRENT_LIST_DIR}
)
