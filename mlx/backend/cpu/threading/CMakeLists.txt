# The CPU thread pool, used only by builds with Highway kernels. The parent adds
# this directory inside its MLX_USE_HIGHWAY block.
target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/openblas/thread_pool.cpp)
