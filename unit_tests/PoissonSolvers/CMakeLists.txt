# Get the relative path for better logging
file(RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding unit tests found in ${_relPath}")

# Common include and link directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

# Helper function to add a test with GTest
function(add_test_with_gtest target source_file)
    add_executable(${target} ${source_file})
    target_link_libraries(${target}
        ippl
        pthread
        GTest::gtest_main
        ${MPI_CXX_LIBRARIES}
    )
    gtest_discover_tests(${target} PROPERTIES TEST_DISCOVERY_TIMEOUT 600)
endfunction()

add_test_with_gtest(feynman_kac_test FeynmanKacSolverTest.cpp)
