# run_hls.tcl
open_project dataflow
set_top forward
add_files ./code/src/forward.cpp
add_files -tb ./code/testbench/testbench.cpp
add_files -tb ./code/testbench/tokenizer.bin
add_files -tb ./code/testbench/model.bin

open_solution "HLS_Dataflow_Synth_20251024" -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period 4 -name default

# Configure interface settings for 2024.2
config_interface -m_axi_addr64=true
config_interface -m_axi_alignment_byte_size=64

csim_design   
csynth_design  
#cosim_design -trace_level none
config_export -format xo -output solution.xo

export_design -format xo

exit
