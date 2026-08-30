# Collect source files
file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS *.cppm)
list(FILTER SRCS EXCLUDE REGEX ".*/bin/.*")

# Add the executable
add_library(evosim_lib STATIC)
target_sources(evosim_lib PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES ${SRCS})
target_link_libraries(evosim_lib PRIVATE spdlog::spdlog)

set(EVOSIM_LINK_LIBS spdlog::spdlog evosim_lib)
set(EVOSIM_LINK_DIRS ${Boost_INCLUDE_DIRS})

# add_executable(repro)  # deprecated old binary, use new modular path via memetic and parity for testing
# # target_sources(repro PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/reproduction.cppm)
# target_link_libraries(repro PRIVATE ${EVOSIM_LINK_LIBS})
# target_link_directories(repro PRIVATE ${EVOSIM_LINK_DIRS})

# add_executable(modulate)  # deprecated
# # target_sources(modulate PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/modulate_initialization.cppm)
# target_link_libraries(modulate PRIVATE ${EVOSIM_LINK_LIBS})
# target_link_directories(modulate PRIVATE ${EVOSIM_LINK_DIRS})

# add_executable(scatter)  # deprecated
# # target_sources(scatter PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/scatter.cppm)
# target_link_libraries(scatter PRIVATE ${EVOSIM_LINK_LIBS})
# target_link_directories(scatter PRIVATE ${EVOSIM_LINK_DIRS})

# add_executable(mitigation)  # deprecated
# # target_sources(mitigation PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/mitigation.cppm)
# target_link_libraries(mitigation PRIVATE ${EVOSIM_LINK_LIBS})
# target_link_directories(mitigation PRIVATE ${EVOSIM_LINK_DIRS})

add_executable(parity)
target_sources(parity PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/parity.cppm)
target_link_libraries(parity PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(parity PRIVATE ${EVOSIM_LINK_DIRS})

# add_executable(memetic)  # new memetic binary, temporarily not built due to template complexity, can be fixed later
# # target_sources(memetic PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/memetic.cppm)
# target_link_libraries(memetic PRIVATE ${EVOSIM_LINK_LIBS})
# target_link_directories(memetic PRIVATE ${EVOSIM_LINK_DIRS})

add_executable(queue_benchmark)
target_sources(queue_benchmark PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/queue_benchmark.cppm)
target_link_libraries(queue_benchmark PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(queue_benchmark PRIVATE ${EVOSIM_LINK_DIRS})

add_executable(gauss_benchmark)
target_sources(gauss_benchmark PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/gauss_benchmark.cppm)
target_link_libraries(gauss_benchmark PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(gauss_benchmark PRIVATE ${EVOSIM_LINK_DIRS})

add_executable(dist_benchmark)
target_sources(dist_benchmark PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/dist_benchmark.cppm)
target_link_libraries(dist_benchmark PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(dist_benchmark PRIVATE ${EVOSIM_LINK_DIRS})

add_executable(uniform_benchmark)
target_sources(uniform_benchmark PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/uniform_benchmark.cppm)
target_link_libraries(uniform_benchmark PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(uniform_benchmark PRIVATE ${EVOSIM_LINK_DIRS})

add_executable(profile_hotspot)
target_sources(profile_hotspot PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/profile_hotspot.cppm)
target_link_libraries(profile_hotspot PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(profile_hotspot PRIVATE ${EVOSIM_LINK_DIRS})

add_executable(comprehensive_benchmark)
target_sources(comprehensive_benchmark PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/comprehensive_benchmark.cppm)
target_link_libraries(comprehensive_benchmark PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(comprehensive_benchmark PRIVATE ${EVOSIM_LINK_DIRS})

add_executable(rng_benchmark)
target_sources(rng_benchmark PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/rng_benchmark.cppm)
target_link_libraries(rng_benchmark PRIVATE ${EVOSIM_LINK_LIBS})
target_link_directories(rng_benchmark PRIVATE ${EVOSIM_LINK_DIRS})
# # add_executable(repro)  # deprecated old binary, use new modular path via memetic and parity for testing
# # # target_sources(repro PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES bin/reproduction.cppm)
# # target_link_libraries(repro PRIVATE ${EVOSIM_LINK_LIBS})
# # target_link_directories(repro PRIVATE ${EVOSIM_LINK_DIRS})
