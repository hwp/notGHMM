# Author : Weipeng He <heweipeng@gmail.com>
# Copyright (c) 2014, All rights reserved.

add_executable(genseq_test genseq_test.c notghmm.c utils.c)
target_link_libraries(genseq_test ${GSL_LIBRARIES})
if(UNIX)
  target_link_libraries(genseq_test m)
endif(UNIX)

