import numpy as np
from PIL import Image
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    return img_array

def transfer_to_gpu(image_array):
    height, width, channels = image_array.shape
    d_image = cuda.mem_alloc(image_array.nbytes)
    cuda.memcpy_htod(d_image, image_array)
    return d_image, (height, width, channels)

grayscale_kernel = """
__global__ void grayscale(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float r = input[idx];
        float g = input[idx + 1];
        float b = input[idx + 2];
        output[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}
"""

mean_intensity_kernel = """
__global__ void calculate_mean_intensity(float *input, float *mean, int width, int height) {
    __shared__ float shared_data[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = (idx < width * height * 3) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(mean, shared_data[0]);
    }
}
"""

brightness_adjustment_kernel = """
__global__ void adjust_brightness(float *input, float *output, float mean, float factor, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        for (int i = 0; i < 3; i++) {
            float val = (input[idx + i] - mean) * factor + mean;
            output[idx + i] = min(max(val, 0.0f), 255.0f);
        }
    }
}
"""

gaussian_blur_kernel = """
__global__ void gaussian_blur(float *input, float *output, int width, int height, int channels, float *kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int half_k = kernel_size / 2;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int ky = -half_k; ky <= half_k; ky++) {
                for (int kx = -half_k; kx <= half_k; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;

                    // Handle edges with padding (mirror)
                    if (nx < 0) nx = -nx - 1;
                    if (ny < 0) ny = -ny - 1;
                    if (nx >= width) nx = 2 * width - nx - 2;
                    if (ny >= height) ny = 2 * height - ny - 2;

                    int nidx = (ny * width + nx) * channels + c;
                    int kidx = (ky + half_k) * kernel_size + (kx + half_k);
                    sum += input[nidx] * kernel[kidx];
                }
            }
            int idx = (y * width + x) * channels + c;
            output[idx] = sum;
        }
    }
}
"""

def run_grayscale(image_array):
    mod = SourceModule(grayscale_kernel)
    grayscale = mod.get_function("grayscale")

    height, width, _ = image_array.shape
    d_input, shape = transfer_to_gpu(image_array)
    d_output = cuda.mem_alloc(height * width * 4)

    block = (16, 16, 1)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])

    grayscale(d_input, d_output, np.int32(width), np.int32(height), block=block, grid=grid)

    output = np.empty((height, width), dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)
    return output.astype(np.uint8)

def run_brightness(image_array, factor):
    height, width, channels = image_array.shape
    mod = SourceModule(mean_intensity_kernel + brightness_adjustment_kernel)
    calculate_mean_intensity = mod.get_function("calculate_mean_intensity")
    adjust_brightness = mod.get_function("adjust_brightness")

    d_input, shape = transfer_to_gpu(image_array)
    d_output = cuda.mem_alloc(image_array.nbytes)
    d_mean = cuda.mem_alloc(4)
    cuda.memset_d32(d_mean, 0, 1)

    block_size = 256
    grid_size = (width * height * channels + block_size - 1) // block_size

    calculate_mean_intensity(d_input, d_mean, np.int32(width), np.int32(height), block=(block_size, 1, 1), grid=(grid_size, 1, 1))

    mean_intensity = np.zeros(1, dtype=np.float32)
    cuda.memcpy_dtoh(mean_intensity, d_mean)
    mean_intensity /= (width * height * channels)

    block = (16, 16, 1)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])

    adjust_brightness(d_input, d_output, np.float32(mean_intensity[0]), np.float32(factor), np.int32(width), np.int32(height), block=block, grid=grid)

    output = np.empty_like(image_array)
    cuda.memcpy_dtoh(output, d_output)
    return output.astype(np.uint8)

def run_gaussian_blur(image_array, kernel_size=5, sigma=1.0):
    kernel = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2 / sigma ** 2)
    kernel = kernel / kernel.sum()
    kernel = np.outer(kernel, kernel)
    kernel = kernel.astype(np.float32)

    mod = SourceModule(gaussian_blur_kernel)
    gaussian_blur = mod.get_function("gaussian_blur")

    height, width, channels = image_array.shape
    d_input, shape = transfer_to_gpu(image_array)
    d_output = cuda.mem_alloc(image_array.nbytes)
    d_kernel = cuda.mem_alloc(kernel.nbytes)
    cuda.memcpy_htod(d_kernel, kernel)

    block = (16, 16, 1)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])

    gaussian_blur(d_input, d_output, np.int32(width), np.int32(height), np.int32(channels), d_kernel, np.int32(kernel_size), block=block, grid=grid)

    output = np.empty_like(image_array)
    cuda.memcpy_dtoh(output, d_output)
    return output.astype(np.uint8)

if __name__ == "__main__":
    image_path = "/content/input.jpeg"
    img = load_image(image_path)

    # Grayscale
    grayscale_img = run_grayscale(img)
    Image.fromarray(grayscale_img).save("grayscale_output.jpg")

    # Brightness adjustment
    bright_img = run_brightness(img, factor=1.2)
    Image.fromarray(bright_img).save("brightness_output.jpg")

    # Gaussian blur
    blurred_img = run_gaussian_blur(img, kernel_size=11, sigma=3.0)
    Image.fromarray(blurred_img).save("blurred_output.jpg")