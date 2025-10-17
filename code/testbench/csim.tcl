# run_hls.tcl
open_project proj_forward
set_top forward
add_files ../src/forward.cpp
add_files -tb testbench.cpp
add_files -tb tokenizer.bin
add_files -tb model.bin

open_solution "sol1"
set_part {xcu50-fsvh2104-2-e}
create_clock -period 4 -name default

# Configure interface settings for 2024.2
config_interface -m_axi_addr64=true
config_interface -m_axi_alignment_byte_size=64

csim_design   
csynth_design  
#cosim_design -trace_level none
export_design -format xo
exit
