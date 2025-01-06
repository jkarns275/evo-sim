# Collect source files
file(GLOB SRCS *.cppm)

# Add the executable
add_library(evosim_lib STATIC)
target_sources(evosim_lib PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES ${SRCS})
target_link_libraries(evosim_lib PRIVATE spdlog::spdlog)

add_executable(evosim)
target_sources(evosim PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/main.cppm)
target_link_libraries(evosim PRIVATE spdlog::spdlog)
target_link_libraries(evosim PRIVATE evosim_lib)
target_link_directories(evosim PRIVATE ${Boost_INCLUDE_DIRS})
