# CUDA Implementation Details

## 1. Define error handler macro

   ![GPU error handler macro code snippet image](./scrots/01.png)

## 2. Define kernel

   ![kernel definition code snippet image](./scrots/02.png)

## 3. Allocate space on GPU for input image

   ![GPU space allocation for input image code snippet image](./scrots/03.png)

## 4. Copy input image from RAM to GPU DRAM

   ![copy input image to GPU code snippet image](./scrots/04.png)

## 5. Allocate space on GPU for kernel arguments

   ![allocate space on GPU for kernel arguments code snippet image](./scrots/05.png)

## 6. Copy kernel arguments to GPU DRAM

   ![copy kernel arguments to GPU code snippet image](./scrots/06.png)

## 7. Allocate space on GPU for output image

   ![GPU space allocation for output image code snippet image](./scrots/07.png)

## 8. Launch the kernel

   ![kernel launch code snippet image](./scrots/08.png)

## 9. Handle kernel launch error

   ![kernel launch code snippet image](./scrots/09.png)

## 10. Wait for GPU to finish

   ![GPU space allocation for input image code snippet image](./scrots/10.png)

## 11. Copy the result from GPU DRAM to RAM

   ![GPU space allocation for input image code snippet image](./scrots/11.png)
