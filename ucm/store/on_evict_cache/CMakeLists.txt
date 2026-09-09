file(GLOB_RECURSE UCM_ON_EVICT_CACHE_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(onevictcachestore SHARED ${UCM_ON_EVICT_CACHE_STORE_CC_SOURCE_FILES})
target_include_directories(onevictcachestore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(onevictcachestore PUBLIC storeintf metrics trans infra_logger)

file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS onevictcachestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
