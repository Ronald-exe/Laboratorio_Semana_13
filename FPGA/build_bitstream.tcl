# build_bitstream.tcl - Arma el diseno de bloques (PS Zynq MPSoC + IP softmax_hls) y genera
# el bitstream (.bit) y el archivo de descripcion de hardware (.hwh) que usa PYNQ.
#
# Requisito: haber corrido run_hls.tcl (exporta el IP de la variante EXPORTAR, por defecto 2).
# Uso, desde la carpeta FPGA/:
#     vivado -mode batch -source build_bitstream.tcl -tclargs fp32
#     vivado -mode batch -source build_bitstream.tcl -tclargs fp16
# (argumento opcional 2: la variante, por defecto 2)
#
# Resultado: overlay/softmax_<prec>.bit y overlay/softmax_<prec>.hwh (mismo nombre base:
# PYNQ busca el .hwh junto al .bit).

set prec [expr {[llength $argv] > 0 ? [lindex $argv 0] : "fp32"}]
set var  [expr {[llength $argv] > 1 ? [lindex $argv 1] : 2}]
set dir  [file dirname [file normalize [info script]]]
set ipdir $dir/proy_${prec}_v${var}/sol/impl/ip
set nombre kv260_softmax_$prec
set pdir $dir/vivado_$prec

if {![file exists $ipdir]} { error "No existe $ipdir: primero corra run_hls.tcl" }

create_project -force $nombre $pdir -part xck26-sfvc784-2LV-c
# Si estan instalados los board files de la KV260 los usamos (configuran el PS como la placa)
set bp [get_board_parts -quiet -latest_file_version *kv260_som*]
if {$bp ne ""} {
    set_property board_part [lindex $bp end] [current_project]
} else {
    puts "AVISO: no hay board files de KV260; se usa solo el part xck26 (PYNQ igual lo carga)"
}
set_property ip_repo_paths $ipdir [current_project]
update_ip_catalog

create_bd_design design_1

# --- Procesador (PS) ---
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
if {$bp ne ""} {
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
}
# HPM0_FPD: el PS (host) escribe los registros AXI-Lite del IP (n, direcciones, start/done)
# HP0_FPD : el IP (m_axi) lee x y escribe y en la DDR
# pl_clk0 : reloj de 100 MHz para el IP (el mismo periodo de 10 ns de la sintesis HLS)
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
] $ps

# --- Nuestro IP ---
set vlnv [lindex [get_ipdefs -all *softmax_hls*] 0]
create_bd_cell -type ip -vlnv $vlnv softmax_hls_0

# --- Conexiones AXI (Vivado agrega SmartConnect y reset automaticamente) ---
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/softmax_hls_0/s_axi_control} ddr_seg {Auto} \
    intc_ip {New AXI SmartConnect} master_apm {0}} [get_bd_intf_pins softmax_hls_0/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/softmax_hls_0/m_axi_gmem0} Slave {/zynq_ultra_ps_e_0/S_AXI_HP0_FPD} ddr_seg {Auto} \
    intc_ip {New AXI SmartConnect} master_apm {0}} [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/softmax_hls_0/m_axi_gmem1} Slave {/zynq_ultra_ps_e_0/S_AXI_HP0_FPD} ddr_seg {Auto} \
    intc_ip {Auto} master_apm {0}} [get_bd_intf_pins softmax_hls_0/m_axi_gmem1]

assign_bd_address
validate_bd_design
save_bd_design

make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "La implementacion fallo" }

file mkdir $dir/overlay
file copy -force $pdir/$nombre.runs/impl_1/design_1_wrapper.bit $dir/overlay/softmax_$prec.bit
set hwh [lindex [glob -nocomplain $pdir/$nombre.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh \
                                  $pdir/$nombre.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh] 0]
file copy -force $hwh $dir/overlay/softmax_$prec.hwh
# Reporte de utilizacion y timing del diseno completo (sirve para el informe)
open_run impl_1
report_utilization -file $dir/overlay/utilizacion_$prec.rpt
report_timing_summary -file $dir/overlay/timing_$prec.rpt
report_power -file $dir/overlay/potencia_estimada_$prec.rpt
puts "LISTO: overlay/softmax_$prec.bit y overlay/softmax_$prec.hwh"
exit
