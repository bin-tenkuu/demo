package demo.md5;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import lombok.val;

/**
 * @author bin
 * @since 2025/07/10
 */
public class JCudaMD5 {
    public static void main(String[] args) {
        val start = System.currentTimeMillis();
        testMd5();
        System.out.printf("Execution time: %d ms%n", System.currentTimeMillis() - start);
    }

    private static void testMd5() {
        // 创建CUDA上下文
        JCuda.setExceptionsEnabled(true);
        JCuda.cudaSetDevice(0);
        // 编译核函数
        val module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, "test/src/main/java/demo/md5/md5.ptx");
        // 获取核函数句柄
        val md5Kernel = new CUfunction();
        JCudaDriver.cuModuleGetFunction(md5Kernel, module, "md5");
        // 创建CUDA流
        val cUstream = new cudaStream_t();
        JCuda.cudaStreamCreate(cUstream);
        // 计算网格和块大小
        int blockSize = 1024; // 每个块1024个线程
        int gridSize = 4; // 足够的块覆盖所有元素
        int size = 3; // 总数据大小
        int[] in = new int[size]; // 输入数据
        // int[] out = new int[size]; // 输入数据
        val cudaSize = (long) size * Sizeof.INT;
        val pIn = new CUdeviceptr();
        // val pOut = new CUdeviceptr();
        JCuda.cudaMalloc(pIn, cudaSize);
        // JCuda.cudaMalloc(pOut, cudaSize);
        // 设置内核参数
        val kernelParameters = Pointer.to(
                Pointer.to(pIn)
                // Pointer.to(pOut),
                // Pointer.to(new int[]{blockSize})
        );
        {
            // JCuda.cudaMemcpyAsync(pIn, Pointer.to(in), cudaSize,
            //         cudaMemcpyKind.cudaMemcpyHostToDevice, cUstream);
            val start = System.currentTimeMillis();
            JCudaDriver.cuLaunchKernel(md5Kernel,
                    gridSize, 1, 1,     // 网格维度
                    blockSize, 1, 1,    // 块维度
                    0, new CUstream(cUstream),            // 共享内存和流
                    kernelParameters, null // 内核参数
            );
            // 将结果复制回主机
            // JCuda.cudaMemcpyAsync(Pointer.to(out), pOut, cudaSize,
            //         cudaMemcpyKind.cudaMemcpyDeviceToHost, cUstream);
            JCuda.cudaLaunchHostFunc(cUstream, time -> {
                System.out.println(
                        "Kernel execution completed in " + (System.currentTimeMillis() - (long) time) + " ms");
            }, start);
        }
        // 等待内核完成
        JCuda.cudaStreamSynchronize(cUstream);
        JCuda.cudaStreamDestroy(cUstream);
        // 释放设备内存
        JCuda.cudaFree(pIn);
        // JCuda.cudaFree(pOut);
        JCudaDriver.cuModuleUnload(module);
    }
}
