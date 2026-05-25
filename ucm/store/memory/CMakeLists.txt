file(GLOB_RECURSE UCM_MEMORY_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(memorystore STATIC ${UCM_MEMORY_STORE_CC_SOURCE_FILES})
target_include_directories(memorystore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc
)
target_link_libraries(memorystore PUBLIC storeintf trans infra_logger)

file(GLOB_RECURSE UCM_MEMORY_STORE_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmmemorystore ${UCM_MEMORY_STORE_CPY_SOURCE_FILES})
target_link_libraries(ucmmemorystore PRIVATE memorystore)

file(RELATIVE_PATH INSTALL_REL_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ucmmemorystore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
