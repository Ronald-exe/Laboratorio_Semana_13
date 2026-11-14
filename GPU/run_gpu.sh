#!/usr/bin/env bash
# run_gpu.sh - Corre y perfila softmax_cuda (FA) en la Jetson Nano. Uso: ./run_gpu.sh
#
# Por cada precision (fp16, fp32) y N (10, 100, 500, 1000):
#   1) ./softmax_cuda N prec MED --csv ...  -> MED mediciones de kernel (cudaEvent) y end-to-end
#      (copia H->D + lanzamiento + sincronizacion + copia D->H, con chrono). El programa ya hace
#      su propia corrida de calentamiento antes de medir.
#   2) nvprof sobre la misma ejecucion -> tiempos del perfilador para el kernel y para las
#      copias de memoria (resumen GPU) y para las llamadas de la API (cudaMemcpy, launch, sync).
# tegrastats corre en paralelo con marca de tiempo epoch (temperatura, frecuencia, carga GPU).
# Variables opcionales: MED=1000
set -u
cd "$(dirname "$0")"
export PATH=/usr/local/cuda/bin:$PATH
MED=${MED:-1000}
RES=resultados; PERF=perfilador
mkdir -p "$RES" "$PERF"

echo ">> Compilando"
nvcc -O3 -std=c++14 softmax_cuda.cu -o softmax_cuda || exit 1

echo ">> Registrando entorno"
{
  echo "fecha: $(date -Is)"
  echo "--- nvpmodel -q"; nvpmodel -q 2>&1 | grep -v NVPM
  echo "--- nucleos CPU online"; cat /sys/devices/system/cpu/online
  echo "--- frecuencia GPU actual/max (Hz)"
  cat /sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq 2>/dev/null
  echo "--- nvcc"; nvcc --version | tail -1
  echo "--- modelo"; tr -d '\0' < /proc/device-tree/model; echo
  echo "--- parametros: MED=$MED"
} > "$RES/entorno.txt" 2>&1

( stdbuf -oL tegrastats --interval 500 | while IFS= read -r l; do echo "$(date +%s.%N) $l"; done ) > "$RES/tegrastats.log" 2>&1 &
TPID=$!
trap 'kill $TPID 2>/dev/null; pkill -x tegrastats 2>/dev/null' EXIT
sleep 2

for P in fp16 fp32; do
  for N in 10 100 500 1000; do
    echo ">> $P N=$N"
    e0=$(date +%s.%N)
    ./softmax_cuda $N $P $MED --csv "$RES/gpu_${P}_n${N}.csv" > "$RES/salida_${P}_n${N}.txt" 2>&1
    e1=$(date +%s.%N)
    echo "$P,$N,$e0,$e1" >> "$RES/ventanas_epoch.csv"
    nvprof --print-gpu-summary --print-api-summary \
      ./softmax_cuda $N $P $MED --csv "$PERF/descartar.csv" > "$PERF/nvprof_${P}_n${N}.txt" 2>&1
    sleep 2
  done
done
rm -f "$PERF/descartar.csv"
echo ">> LISTO"
