if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    if(APPLE)
        file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/memory/*.cc")
    else()
        file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    endif()
    add_executable(ucmstore.test ${UCMSTORE_TEST_SOURCE_FILES})
    target_include_directories(ucmstore.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    if(APPLE)
        target_link_libraries(ucmstore.test PRIVATE
            memorystore
            gtest_main gtest gmock
        )
    else()
        target_link_libraries(ucmstore.test PRIVATE
            posixstore cachestore memorystore fakestore pcstore
            gtest_main gtest gmock
        )
    endif()
    gtest_discover_tests(ucmstore.test)
endif()
