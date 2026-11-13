#!/usr/bin/env bash
# Ejecuta TODO el protocolo CPU en la Jetson Nano. Uso: ./run_all.sh
# Variables opcionales: REPS=10000 RUNS=5 SEED=12345 SEG=10 CORE=1 (fija el proceso a un nucleo)
set -uo pipefail
cd "$(dirname "$0")"
RES=resultados; mkdir -p "$RES"
REPS=${REPS:-10000}; RUNS=${RUNS:-5}; SEED=${SEED:-12345}; SEG=${SEG:-10}
SIZES="10 100 500 1000"
PIN=""; [ -n "${CORE:-}" ] && PIN="taskset -c $CORE"

echo ">> Compilando"
g++ -O3 -std=c++17 -march=native softmax_cpu.cpp -o softmax_cpu -lm || exit 1

echo ">> Registrando entorno en $RES/entorno.txt"
{
  echo "fecha: $(date -Is)"; echo "--- g++ -v"; g++ --version | head -1
  echo "--- uname"; uname -a
  echo "--- nvpmodel -q"; sudo nvpmodel -q 2>&1 || true
  echo "--- jetson_clocks --show"; sudo jetson_clocks --show 2>&1 || true
  echo "--- nucleos online"; cat /sys/devices/system/cpu/online
  echo "--- governor / frecuencia actual (kHz)"
  for c in /sys/devices/system/cpu/cpu[0-9]*; do
    echo "$(basename $c): $(cat $c/cpufreq/scaling_governor 2>/dev/null) $(cat $c/cpufreq/scaling_cur_freq 2>/dev/null)"
  done
  echo "--- lscpu"; lscpu
  echo "--- parametros: REPS=$REPS RUNS=$RUNS SEED=$SEED SEG=$SEG CORE=${CORE:-sin fijar}"
} > "$RES/entorno.txt" 2>&1

# tegrastats con marca de tiempo epoch por linea (para cruzar con epoch_inicio/epoch_fin)
TPID=""
if command -v tegrastats >/dev/null; then
  ( stdbuf -oL tegrastats --interval 500 | while IFS= read -r l; do echo "$(date +%s.%N) $l"; done ) > "$RES/tegrastats.log" 2>&1 &
  TPID=$!
else
  echo "AVISO: tegrastats no encontrado; no se registrara potencia/temperatura"
fi
trap '[ -n "$TPID" ] && kill $TPID 2>/dev/null; pkill -x tegrastats 2>/dev/null' EXIT
sleep 2

for P in fp16 fp32; do
  echo ">> Validacion $P"
  first=1
  for N in $SIZES; do
    H=""; [ $first -eq 1 ] && H="--header"; first=0
    $PIN ./softmax_cpu --precision $P --n $N --mode validar --seed $SEED $H 2>/dev/null >> "$RES/validacion_$P.csv.tmp"
  done
  mv "$RES/validacion_$P.csv.tmp" "$RES/validacion_$P.csv"

  echo ">> Mediciones $P (funcion + programa completo)"
  OUT="$RES/mediciones_$P.csv"
  echo "precision,n,reps,run,t_funcion_us_por_llamada,epoch_inicio,epoch_fin,t_programa_completo_ms" > "$OUT"
  for N in $SIZES; do
    $PIN ./softmax_cpu --precision $P --n $N --mode medir --reps $REPS --runs 1 --seed $SEED >/dev/null 2>&1  # calentamiento descartado
    for R in $(seq 1 $RUNS); do
      t0=$(date +%s%N)
      line=$($PIN ./softmax_cpu --precision $P --n $N --mode medir --reps $REPS --runs 1 --run-id $R --seed $SEED 2>/dev/null)
      t1=$(date +%s%N)
      ms=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.3f",(b-a)/1e6}')
      echo "$line,$ms" >> "$OUT"
    done
  done

  echo ">> Ventana larga para potencia/energia $P (${SEG}s por N)"
  OUTP="$RES/potencia_$P.csv"
  echo "precision,n,llamadas,segundos,t_llamada_us,epoch_inicio,epoch_fin" > "$OUTP"
  for N in $SIZES; do
    $PIN ./softmax_cpu --precision $P --n $N --mode potencia --segundos $SEG --seed $SEED 2>/dev/null >> "$OUTP"
    sleep 3   # deja enfriar/estabilizar entre configuraciones
  done
done
echo ">> Listo. Resumen: python3 resumen.py"
