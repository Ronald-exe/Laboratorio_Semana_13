#!/usr/bin/env bash
# Corre solo la parte de nvprof (requiere root en Jetson por ERR_NVGPUCTRPERM)
cd "$(dirname "$0")"
for P in fp16 fp32; do for N in 10 100 500 1000; do
  /usr/local/cuda/bin/nvprof --print-gpu-summary --print-api-summary ./softmax_cuda $N $P 1000 --csv /tmp/descartar_nvprof.csv > perfilador/nvprof_${P}_n${N}.txt 2>&1
done; done
rm -f /tmp/descartar_nvprof.csv
chown -R el5859:el5859 perfilador
echo LISTO
