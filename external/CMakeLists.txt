project(external C CXX)

set(TINYEXR_USE_MINIZ ON CACHE BOOL "TINYEXR_USE_MINIZ" FORCE)
add_library(tinyexr STATIC tinyexr/tinyexr.h tinyexr/deps/miniz/miniz.h tinyexr/deps/miniz/miniz.c)
target_include_directories(tinyexr SYSTEM PUBLIC tinyexr tinyexr/deps/miniz)
add_library(ext::tinyexr ALIAS tinyexr)
