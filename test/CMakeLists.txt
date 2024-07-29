include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

file(GLOB_RECURSE KUIPER_TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*/test*.cpp")

foreach (kuiper_test_source ${KUIPER_TEST_SOURCES})
    # Create a human readable name.
    get_filename_component(kuiper_test_filename ${kuiper_test_source} NAME)
    string(REPLACE ".cpp" "" kuiper_test_name ${kuiper_test_filename})

    # Add the test target separately and as part of "make check-tests".
    add_executable(${kuiper_test_name} EXCLUDE_FROM_ALL ${kuiper_test_source})

    target_include_directories(${kuiper_test_name} PUBLIC ../kuiper/include)
    target_link_directories(${kuiper_test_name} PUBLIC ${PROJECT_SOURCE_DIR}/lib)
    target_link_libraries(${kuiper_test_name} llama glog GTest::Main )

    # Set test target properties and dependencies.
    set_target_properties(${kuiper_test_name}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
            COMMAND ${kuiper_test_name}
            )
endforeach ()
