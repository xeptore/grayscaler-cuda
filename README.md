# grayscaler

Educational-purpose multi-threaded image grayscaler.

Currently, only jpeg images are supported using `libjpeg`.

## Usage

1. Configure

   You can configure

      - input image name (`INPUT_IMAGE_FILENAME`)
      - output images name (`OUTPUT_IMAGE_FILENAME`)
      - number of threads in each block (`BLOCK_THREADS`)
      - number of blocks in each grid (`GRID_BLOCKS`)

   configuration parameters in [`config.h`](/config.h) file.

2. Build

   ```sh
   make build
   ```

3. Run

   ```sh
   ./build/grayscale
   ```

## Development

You'll need CMake and a C compiler. I used CMake version `3.20.5` and clang version `12.0.1`.
