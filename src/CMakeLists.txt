# Collect source files
file(GLOB SRCS *.cppm)

# Add the executable
add_library(evosim_lib STATIC)
target_sources(evosim_lib PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES ${SRCS})
target_link_libraries(evosim_lib PRIVATE spdlog::spdlog)

set(EVOSIM_LINK_LIBS spdlog::spdlog evosim_lib)
set(EVOSIM_LINK_DIRS ${Boost_INCLUDE_DIRS})

add_executable(repro)
target_sources(repro PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/reproduction.cppm)
target_link_libraries(repro PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(repro PRIVATE ${EVOSIM_LINK_DIRS})

add_executable(modulate)
target_sources(modulate PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/modulate_initialization.cppm)
target_link_libraries(modulate PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(modulate PRIVATE ${EVOSIM_LINK_DIRS})

add_executable(scatter)
target_sources(scatter PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/scatter.cppm)
target_link_libraries(scatter PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(scatter PRIVATE ${EVOSIM_LINK_DIRS})

# add_executable(repro)
# target_sources(repro PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/reproduction.cppm)
# target_link_libraries(repro PRIVATE ${EVOSIM_LINK_LIBS})
# target_link_directories(repro PRIVATE ${EVOSIM_LINK_DIRS})
