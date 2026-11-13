#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <math_constants.h>

uint64_t splitmix64(uint64_t& estado) {
    estado += 0x9E3779B97F4A7C15ULL;
    uint64_t z = estado;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ float load_value(const float* value) {
    return *value;
}

__device__ float load_value(const __half* value) {
    return __half2float(*value);
}

__device__ void store_value(float* destination, float value) {
    *destination = value;
}

__device__ void store_value(__half* destination, float value) {
    *destination = __float2half(value);
}

template <typename T>
__global__ void softmax_kernel(const T* x, T* y, int n) {
    int tid = threadIdx.x;
    float local_max = -CUDART_INF_F;

    for (int i = tid; i < n; i += blockDim.x) {
        float value = load_value(x + i);
        if (value > local_max) {
            local_max = value;
        }
    }

    __shared__ float maximos[256];
    maximos[tid] = local_max;
    __syncthreads();

    for (int salto = 128; salto > 0; salto /= 2) {
        if (tid < salto && maximos[tid + salto] > maximos[tid]) {
            maximos[tid] = maximos[tid + salto];
        }
        __syncthreads();
    }

    // Keep the exponential reduction in float for both FP32 and FP16 outputs.
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float exponential = expf(load_value(x + i) - maximos[0]);
        local_sum += exponential;
    }

    __shared__ float sumas[256];
    sumas[tid] = local_sum;
    __syncthreads();

    for (int salto = 128; salto > 0; salto /= 2) {
        if (tid < salto) {
            sumas[tid] += sumas[tid + salto];
        }
        __syncthreads();
    }

    for (int i = tid; i < n; i += blockDim.x) {
        float exponential = expf(load_value(x + i) - maximos[0]);
        store_value(y + i, exponential / sumas[0]);
    }
}

static bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << ": " << cudaGetErrorString(status) << std::endl;
    return false;
}

static bool parse_integer(const char* text, int& value) {
    try {
        size_t pos = 0;
        int parsed = std::stoi(text, &pos);
        if (pos != std::strlen(text)) {
            return false;
        }
        value = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

static void launch_softmax(bool fp16, const void* x, void* y, int n) {
    if (fp16) {
        softmax_kernel<<<1, 256>>>(
            static_cast<const __half*>(x), static_cast<__half*>(y), n);
    } else {
        softmax_kernel<<<1, 256>>>(
            static_cast<const float*>(x), static_cast<float*>(y), n);
    }
}

struct Validation {
    double max_abs_error;
    double max_rel_error;
    double output_sum;
    bool range_valid;
    bool sum_valid;
    bool max_index_valid;
};

static Validation validate_output(
    const std::vector<double>& x_original,
    const std::vector<double>& y_ref,
    const std::vector<float>& y,
    bool fp16) {
    int input_max_index = 0;
    int output_max_index = 0;
    double output_sum = 0.0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    bool range_valid = true;

    for (size_t i = 0; i < y.size(); ++i) {
        double value = y[i];
        double abs_error = std::fabs(value - y_ref[i]);
        double rel_error = abs_error / std::fabs(y_ref[i]);
        output_sum += value;

        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }
        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
        }
        if (x_original[i] > x_original[input_max_index]) {
            input_max_index = static_cast<int>(i);
        }
        if (y[i] > y[output_max_index]) {
            output_max_index = static_cast<int>(i);
        }
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
            range_valid = false;
        }
    }

    double sum_tolerance = fp16 ? 1e-2 : 1e-5;
    return {
        max_abs_error,
        max_rel_error,
        output_sum,
        range_valid,
        std::fabs(output_sum - 1.0) < sum_tolerance,
        input_max_index == output_max_index
    };
}

int main(int argc, char** argv) {
    int n = 10;
    bool fp16 = false;
    int repetitions = 5;
    std::string csv_path;

    int arg = 1;
    if (arg < argc && std::strncmp(argv[arg], "--", 2) != 0) {
        if (!parse_integer(argv[arg], n)) {
            std::cerr << "Tamaño inválido: " << argv[arg] << std::endl;
            return 1;
        }
        ++arg;
    }
    if (arg < argc && std::strncmp(argv[arg], "--", 2) != 0 &&
        (std::string(argv[arg]) == "fp32" || std::string(argv[arg]) == "fp16")) {
        fp16 = std::string(argv[arg]) == "fp16";
        ++arg;
    }
    if (arg < argc && std::strncmp(argv[arg], "--", 2) != 0) {
        if (!parse_integer(argv[arg], repetitions)) {
            std::cerr << "Cantidad de repeticiones inválida: " << argv[arg] << std::endl;
            return 1;
        }
        ++arg;
    }

    while (arg < argc) {
        std::string option = argv[arg++];
        if (option == "--csv" && arg < argc) {
            csv_path = argv[arg++];
            if (csv_path.empty()) {
                std::cerr << "La ruta CSV no puede estar vacía." << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Opción desconocida o sin valor: " << option << std::endl;
            return 1;
        }
    }

    if (n != 10 && n != 100 && n != 500 && n != 1000) {
        std::cerr << "n debe ser 10, 100, 500 o 1000" << std::endl;
        return 1;
    }
    if (repetitions < 5) {
        std::cerr << "Se requieren al menos cinco repeticiones." << std::endl;
        return 1;
    }

    const std::string precision = fp16 ? "fp16" : "fp32";
    if (csv_path.empty()) {
        csv_path = "softmax_results_n" + std::to_string(n) + "_" +
                   precision + ".csv";
    }

    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "No se pudo abrir el CSV: " << csv_path << std::endl;
        return 1;
    }
    csv << "tipo,n,precision,semilla,repeticion,kernel_ms,end_to_end_ms,error_abs_max,"
           "error_rel_max,suma,suma_valida,rango_valido,maximo_valido"
           "\n";

    const uint64_t seed = 12345;
    uint64_t state = seed;
    std::vector<double> x_original(n);
    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    std::vector<__half> h_x16;
    std::vector<__half> h_y16;
    if (fp16) {
        h_x16.resize(n);
        h_y16.resize(n);
    }
    for (int i = 0; i < n; ++i) {
        uint64_t random_value = splitmix64(state);
        double u = static_cast<double>(random_value >> 11) *
                   (1.0 / 9007199254740992.0);
        x_original[i] = -4.0 + 8.0 * u;
        h_x[i] = static_cast<float>(x_original[i]);
        if (fp16) {
            h_x16[i] = __float2half_rn(h_x[i]);
        }
    }

    double max_ref = x_original[0];
    for (int i = 1; i < n; ++i) {
        if (x_original[i] > max_ref) {
            max_ref = x_original[i];
        }
    }
    std::vector<double> y_ref(n);
    double reference_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        y_ref[i] = std::exp(x_original[i] - max_ref);
        reference_sum += y_ref[i];
    }
    for (double& value : y_ref) {
        value /= reference_sum;
    }

    size_t bytes = static_cast<size_t>(n) *
                   (fp16 ? sizeof(__half) : sizeof(float));
    void* d_x = nullptr;
    void* d_y = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    auto cleanup = [&]() {
        if (start_event != nullptr) {
            check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
            start_event = nullptr;
        }
        if (stop_event != nullptr) {
            check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");
            stop_event = nullptr;
        }
        if (d_x != nullptr) {
            check_cuda(cudaFree(d_x), "cudaFree(d_x)");
            d_x = nullptr;
        }
        if (d_y != nullptr) {
            check_cuda(cudaFree(d_y), "cudaFree(d_y)");
            d_y = nullptr;
        }
    };

    if (!check_cuda(cudaMalloc(&d_x, bytes), "cudaMalloc(d_x)") ||
        !check_cuda(cudaMalloc(&d_y, bytes), "cudaMalloc(d_y)") ||
        !check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)") ||
        !check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)")) {
        cleanup();
        return 1;
    }

    auto copy_input = [&]() {
        const void* host_input = fp16
            ? static_cast<const void*>(h_x16.data())
            : static_cast<const void*>(h_x.data());
        return cudaMemcpy(d_x, host_input, bytes, cudaMemcpyHostToDevice);
    };
    auto copy_output = [&]() {
        void* host_output = fp16
            ? static_cast<void*>(h_y16.data())
            : static_cast<void*>(h_y.data());
        return cudaMemcpy(host_output, d_y, bytes, cudaMemcpyDeviceToHost);
    };
    auto decode_output = [&]() {
        if (fp16) {
            for (int i = 0; i < n; ++i) {
                h_y[i] = __half2float(h_y16[i]);
            }
        }
    };
    auto run_and_sync = [&]() {
        launch_softmax(fp16, d_x, d_y, n);
        cudaError_t launch_status = cudaGetLastError();
        if (!check_cuda(launch_status, "lanzamiento de softmax")) {
            return false;
        }
        return check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    };

    if (!check_cuda(copy_input(), "copia host a device (calentamiento)") ||
        !run_and_sync() ||
        !check_cuda(copy_output(), "copia device a host (calentamiento)")) {
        cleanup();
        return 1;
    }
    decode_output();

    Validation last_validation{};
    bool validations_passed = true;
    for (int repetition = 1; repetition <= repetitions; ++repetition) {
        if (!check_cuda(copy_input(), "copia host a device")) {
            cleanup();
            return 1;
        }
        if (!check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)")) {
            cleanup();
            return 1;
        }
        launch_softmax(fp16, d_x, d_y, n);
        if (!check_cuda(cudaGetLastError(), "lanzamiento de softmax") ||
            !check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)") ||
            !check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize")) {
            cleanup();
            return 1;
        }
        float kernel_ms = 0.0f;
        if (!check_cuda(cudaEventElapsedTime(&kernel_ms, start_event, stop_event),
                        "cudaEventElapsedTime")) {
            cleanup();
            return 1;
        }

        auto total_start = std::chrono::steady_clock::now();
        if (!check_cuda(copy_input(), "copia host a device (end-to-end)")) {
            cleanup();
            return 1;
        }
        launch_softmax(fp16, d_x, d_y, n);
        if (!check_cuda(cudaGetLastError(), "lanzamiento end-to-end") ||
            !check_cuda(cudaDeviceSynchronize(), "sincronización end-to-end") ||
            !check_cuda(copy_output(), "copia device a host (end-to-end)")) {
            cleanup();
            return 1;
        }
        auto total_end = std::chrono::steady_clock::now();
        decode_output();
        double end_to_end_ms = std::chrono::duration<double, std::milli>(
            total_end - total_start).count();
        last_validation = validate_output(x_original, y_ref, h_y, fp16);
        validations_passed = validations_passed &&
            last_validation.range_valid &&
            last_validation.sum_valid &&
            last_validation.max_index_valid;

        csv << "benchmark," << n << ',' << precision << ',' << seed << ','
            << repetition << ',' << std::setprecision(9) << kernel_ms << ','
            << end_to_end_ms << ',' << last_validation.max_abs_error << ','
            << last_validation.max_rel_error << ',' << last_validation.output_sum
            << ',' << last_validation.sum_valid << ',' << last_validation.range_valid
            << ',' << last_validation.max_index_valid << '\n';
        csv.flush();
        if (!csv) {
            std::cerr << "Error al escribir el CSV: " << csv_path << std::endl;
            cleanup();
            return 1;
        }

        std::cout << "Repetición " << repetition << '/' << repetitions
                  << " - kernel: " << kernel_ms << " ms, end-to-end: "
                  << end_to_end_ms
                  << " ms, error abs. máx.: " << last_validation.max_abs_error
                  << ", error rel. máx.: " << last_validation.max_rel_error
                  << std::endl;
    }

    std::cout << "Suma: " << last_validation.output_sum
              << ", rango válido: " << last_validation.range_valid
              << ", suma válida: " << last_validation.sum_valid
              << ", máximo válido: " << last_validation.max_index_valid
              << std::endl;
    cleanup();
    return validations_passed ? 0 : 2;
}
