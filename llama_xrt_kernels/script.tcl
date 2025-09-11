open_project llama_xrt_kernels
set_top forward
add_files src/forward.cpp
add_files src/typedefs.h
open_solution "solution1"
set_part xcu50-fsvh2104-2-e
create_clock -period 6.25 -name default
csynth_design
export_design -format ip_catalog
exit