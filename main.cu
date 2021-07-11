#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <cuda.h>
#include <memory.h>
#include <jpeglib.h>
#include <pthread.h>
#include "config.h"


const unsigned int NUM_THREADS = BLOCK_THREADS * GRID_BLOCKS;
const unsigned char INPUT_IMAGE_COMPONENTS_NUMBER = 3;
const unsigned char OUTPUT_IMAGE_COMPONENTS_NUMBER = 1;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPU: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ unsigned char calculate_gray(
  const unsigned char red,
  const unsigned char green,
  const unsigned char blue
) {
  return red * 0.2126 + green * 0.7152 + blue * 0.0722;
}

void set_decompressor_options(
  struct jpeg_decompress_struct *decompressor,
  struct jpeg_error_mgr *error_manager,
  FILE *input_file
) {
  decompressor->err = jpeg_std_error(error_manager);
  jpeg_create_decompress(decompressor);
  jpeg_stdio_src(decompressor, input_file);
  (void)jpeg_read_header(decompressor, TRUE);
  (void)jpeg_start_decompress(decompressor);
}

void set_compressor_options(
  struct jpeg_compress_struct *compressor,
  const struct jpeg_decompress_struct *decompressor,
  struct jpeg_error_mgr *error_manager,
  FILE *output_file
) {
  compressor->err = jpeg_std_error(error_manager);
  jpeg_create_compress(compressor);
  jpeg_stdio_dest(compressor, output_file);
  compressor->in_color_space = JCS_GRAYSCALE;
  compressor->input_components = OUTPUT_IMAGE_COMPONENTS_NUMBER;
  jpeg_set_defaults(compressor);
  compressor->image_width = decompressor->output_width;
  compressor->image_height = decompressor->image_height;
  compressor->density_unit = decompressor->density_unit;
  compressor->X_density = decompressor->X_density;
  compressor->Y_density = decompressor->Y_density;
  jpeg_start_compress(compressor, TRUE);
}

size_t calculate_input_image_row_length(
  const struct jpeg_decompress_struct *decompressor
) {
  return decompressor->output_width * decompressor->num_components;
}

struct transform_row_params {
  size_t image_width;
  size_t start_row;
  size_t num_rows;
};

__global__ void transform_rows(
  struct transform_row_params *params,
  unsigned char *input,
  unsigned char *output
) {
  const unsigned long int workerIdx = threadIdx.x + blockDim.x * blockIdx.x;
  const size_t image_width = params[workerIdx].image_width;
  for (size_t i = params[workerIdx].start_row; i < params[workerIdx].start_row + params[workerIdx].num_rows; i++) {
    for (size_t j = 0; j < image_width; j++) {
      const unsigned char red = input[(i * image_width * INPUT_IMAGE_COMPONENTS_NUMBER) + j * INPUT_IMAGE_COMPONENTS_NUMBER + 0];
      const unsigned char green = input[(i * image_width * INPUT_IMAGE_COMPONENTS_NUMBER) + j * INPUT_IMAGE_COMPONENTS_NUMBER + 1];
      const unsigned char blue = input[(i * image_width * INPUT_IMAGE_COMPONENTS_NUMBER) + j * INPUT_IMAGE_COMPONENTS_NUMBER + 2];
      const unsigned char gray = calculate_gray(red, green, blue);
      output[i * image_width + j] = gray;
    }
  }
}

int transform_image(const char *input_filename, const char *output_filename) {
  FILE *input_file = fopen(input_filename, "rb");
  if (!input_file) {
    (void)fprintf(
      stderr,
      "🛑🙁 error opening jpeg file '%s': %s 🙁🛑\n",
      input_filename,
      strerror(errno)
    );
    return errno;
  }

  FILE *output_file = fopen(output_filename, "wb");
  if (!output_file) {
    (void)fprintf(
      stderr,
      "🛑🙁 error opening output jpeg file '%s': %s 🙁🛑\n",
      output_filename,
      strerror(errno)
    );
    return errno;
  }

  struct jpeg_error_mgr error_manager;

  struct jpeg_decompress_struct decompressor;
  set_decompressor_options(&decompressor, &error_manager, input_file);

  if (decompressor.image_height < NUM_THREADS) {
    (void)fprintf(
      stderr,
      "🛑🤔 how is that possible to distribute processing %u rows on %u threads? 🤔🛑\n",
      decompressor.image_height,
      NUM_THREADS
    );
    return 1;

  }

  struct jpeg_compress_struct compressor;
  set_compressor_options(&compressor, &decompressor, &error_manager, output_file);

  const size_t input_image_row_length = calculate_input_image_row_length(&decompressor);
  const unsigned int output_image_row_length = compressor.image_width;

  const size_t IMAGE_HEIGHT = decompressor.image_height;
  const size_t INPUT_IMAGE_SIZE_IN_BYTES = decompressor.image_height * decompressor.image_width * INPUT_IMAGE_COMPONENTS_NUMBER;
  const size_t OUTPUT_IMAGE_SIZE_IN_BYTES = compressor.image_height * compressor.image_width * OUTPUT_IMAGE_COMPONENTS_NUMBER;

  unsigned char *all_buffer = (unsigned char *)malloc(INPUT_IMAGE_SIZE_IN_BYTES + OUTPUT_IMAGE_SIZE_IN_BYTES);
  if (all_buffer == NULL) {
    (void)fprintf(stderr, "failed to allocate enough memory.\n");
    exit(-1);
  }

  unsigned char *input_buffer = &all_buffer[0];

  JSAMPROW scan_rows_buffer[decompressor.image_height];
  for (size_t i = 0; i < decompressor.image_height; i++) {
    scan_rows_buffer[i] = &input_buffer[i * input_image_row_length];
  }

  while (decompressor.output_scanline < decompressor.output_height) {
    (void)jpeg_read_scanlines(
      &decompressor,
      &scan_rows_buffer[decompressor.output_scanline],
      decompressor.output_height - decompressor.output_scanline
    );
  }

  unsigned char *device_input_buffer;
  gpuErrchk(cudaMallocManaged(&device_input_buffer, INPUT_IMAGE_SIZE_IN_BYTES));
  const size_t input_image_width_in_bytes = decompressor.image_width * INPUT_IMAGE_COMPONENTS_NUMBER;
  for (size_t i = 0; i < IMAGE_HEIGHT; i++) {
    for (size_t j = 0; j < input_image_width_in_bytes; j++) {
      device_input_buffer[i * input_image_width_in_bytes + j] = scan_rows_buffer[i][j];
    }
  }

  unsigned char *output_buffer = &all_buffer[INPUT_IMAGE_SIZE_IN_BYTES];
  JSAMPROW output_rows_buffer[compressor.image_height];
  for (size_t i = 0; i < compressor.image_height; i++) {
    output_rows_buffer[i] = &output_buffer[i * output_image_row_length];
  }

  struct transform_row_params thread_params_refs[NUM_THREADS];

  const unsigned int quotient = decompressor.image_height / NUM_THREADS;
  const unsigned int remainder = decompressor.image_height % NUM_THREADS;

  unsigned long total_assigned_rows = 0U;
  for (size_t i = 0; i < NUM_THREADS; i++) {
    const unsigned long int worker_quotient = (i < remainder) ? (quotient + 1) : (quotient);
    thread_params_refs[i].image_width = decompressor.image_width;
    thread_params_refs[i].start_row = total_assigned_rows;
    thread_params_refs[i].num_rows = worker_quotient;
    total_assigned_rows += worker_quotient;
  }

  struct transform_row_params *device_param_refs;
  gpuErrchk(cudaMallocManaged(&device_param_refs, sizeof(struct transform_row_params) * NUM_THREADS));
  gpuErrchk(cudaMemcpy(
    device_param_refs,
    thread_params_refs,
    sizeof(struct transform_row_params) * NUM_THREADS,
    cudaMemcpyHostToDevice
  ));

  unsigned char *device_output_buffer;
  gpuErrchk(cudaMallocManaged(&device_output_buffer, OUTPUT_IMAGE_SIZE_IN_BYTES));

  dim3 grid_size(GRID_BLOCKS, 1, 1);
  dim3 block_size(BLOCK_THREADS, 1, 1);

  struct timespec start, end;
  timespec_get(&start, TIME_UTC);

  transform_rows<<<grid_size, block_size>>>(device_param_refs, device_input_buffer, device_output_buffer);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  timespec_get(&end, TIME_UTC);
  unsigned long int time_in_nano_seconds = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
  printf("%lu\n", time_in_nano_seconds);

  unsigned char *temp = (unsigned char *)malloc(OUTPUT_IMAGE_SIZE_IN_BYTES);
  gpuErrchk(cudaMemcpy(
    temp,
    device_output_buffer,
    OUTPUT_IMAGE_SIZE_IN_BYTES,
    cudaMemcpyDeviceToHost
  ));
  for (size_t i = 0; i < compressor.image_height; i++) {
    for (size_t j = 0; j < compressor.image_width; j++) {
      output_rows_buffer[i][j] = temp[i * compressor.image_width + j];
    }
  }

  for (size_t i = 0; i < compressor.image_height; i++) {
    (void)jpeg_write_scanlines(&compressor, &output_rows_buffer[i], 1);
  }

  (void)jpeg_finish_decompress(&decompressor);
  jpeg_finish_compress(&compressor);
  jpeg_destroy_decompress(&decompressor);
  jpeg_destroy_compress(&compressor);
  free(all_buffer);
  free(temp);
  cudaFree(&device_input_buffer);
  cudaFree(&device_param_refs);
  cudaFree(&device_output_buffer);
  (void)fclose(input_file);
  (void)fclose(output_file);

  return 0;
}

int main() {
  return transform_image(INPUT_IMAGE_FILENAME, OUTPUT_IMAGE_FILENAME);
}
