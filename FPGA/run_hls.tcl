# run_hls.tcl - Corre TODO el flujo HLS: C-sim, sintesis, cosimulacion y export del IP
#
# Uso (Vitis 2024.2 o posterior, desde la carpeta FPGA/):
#     vitis-run --mode hls --tcl run_hls.tcl
# (en Vitis HLS 2023.2 o anterior:  vitis_hls -f run_hls.tcl)
#
# Variables de entorno opcionales:
#     PRECS="fp32 fp16"     precisiones a correr
#     VARIANTES="0 1 2 3 4" variantes a correr
#     COSIM=0               salta la cosimulacion (mas rapido)
#     EXPORTAR=4            variante que se exporta como IP para Vivado
#
# Por cada precision y variante se crea proy_<prec>_v<k>/ y se copian los reportes a
# reportes/<prec>_v<k>/ (csynth.rpt/.xml, cosim y la validacion de la C-sim).

set dir      [file dirname [file normalize [info script]]]
set part     xck26-sfvc784-2LV-c
set periodo  10   ;# ns -> 100 MHz, el mismo reloj pl_clk0 que usamos en el diseno de Vivado

set precs     [expr {[info exists ::env(PRECS)]     ? $::env(PRECS)     : "fp32 fp16"}]
set variantes [expr {[info exists ::env(VARIANTES)] ? $::env(VARIANTES) : "0 1 2 3 4"}]
set do_cosim  [expr {[info exists ::env(COSIM)]     ? $::env(COSIM)     : 1}]
set exportar  [expr {[info exists ::env(EXPORTAR)]  ? $::env(EXPORTAR)  : 4}]

file mkdir $dir/reportes

foreach p $precs {
    foreach v $variantes {
        set cflags "-DVARIANTE=$v"
        if {$p eq "fp16"} { append cflags " -DUSE_FP16" }
        set proy proy_${p}_v${v}
        puts "=================== $proy ($cflags) ==================="

        open_project -reset $proy
        set_top softmax_hls
        add_files     $dir/softmax_hls.cpp -cflags $cflags
        add_files     $dir/softmax_hls.h
        add_files -tb $dir/softmax_tb.cpp  -cflags $cflags
        open_solution -reset sol
        set_part $part
        create_clock -period $periodo -name default

        csim_design
        csynth_design
        if {$do_cosim} {
            if {[catch {cosim_design -rtl verilog} err]} { puts "AVISO: cosim fallo en $proy: $err" }
        }
        if {$v == $exportar} {
            if {[catch {export_design -format ip_catalog -rtl verilog} err]} { puts "AVISO: export fallo en $proy: $err" }
        }

        # Copiamos los reportes a una carpeta corta para el repositorio
        set out $dir/reportes/${p}_v${v}
        file mkdir $out
        foreach f [glob -nocomplain $proy/sol/syn/report/csynth.rpt \
                                    $proy/sol/syn/report/csynth.xml \
                                    $proy/sol/csim/build/validacion_hls.csv] {
            file copy -force $f $out
        }
        if {[file exists $proy/sol/sim/report]} {
            file delete -force $out/cosim
            file copy -force $proy/sol/sim/report $out/cosim
        }
        close_project
    }
}
exit
