add_library(contextstore SHARED cc/context_store.cc cc/context_index.cc cc/trans_buffer.cc cc/load_queue.cc cc/dump_queue.cc)
target_include_directories(contextstore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(contextstore PUBLIC storeintf trans infra_logger metrics pthread rt)
set_target_properties(contextstore PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../shared/metrics"
)
install(TARGETS contextstore LIBRARY DESTINATION ucm/store/context COMPONENT ucm)

if(BUILD_UNIT_TESTS AND RUNTIME_ENVIRONMENT STREQUAL "simu")
    include(GoogleTest)
    add_executable(contextstore.test ../test/case/context/context_store_test.cc)
    target_link_libraries(contextstore.test PRIVATE contextstore gtest_main)
    gtest_discover_tests(contextstore.test)
endif()
