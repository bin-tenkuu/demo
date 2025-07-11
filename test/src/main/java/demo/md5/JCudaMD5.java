package demo.md5;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
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
        int size = 4; // 总数据大小
        int[] in = new int[size]; // 输入数据
        val cudaSize = (long) size * Sizeof.INT;
        val pIn = new CUdeviceptr();
        JCuda.cudaMalloc(pIn, cudaSize);
        // 设置内核参数
        val kernelParameters = Pointer.to(
                Pointer.to(pIn)
        );
        // for (int i = 0; i < 1; i++)
        {
            JCuda.cudaMemcpyAsync(pIn, Pointer.to(in), cudaSize,
                    cudaMemcpyKind.cudaMemcpyHostToDevice, cUstream);
            val start = System.currentTimeMillis();

            JCudaDriver.cuLaunchKernel(md5Kernel,
                    1 << 5, 1, 1,     // 网格维度
                    1024, 1, 1,    // 块维度
                    0, new CUstream(cUstream),            // 共享内存和流
                    kernelParameters, null // 内核参数
            );
            JCuda.cudaLaunchHostFunc(cUstream, time -> {
                System.out.println(
                        "Kernel execution completed in " + (System.currentTimeMillis() - (long) time) + " ms");
            }, start);
            // 等待内核完成
            JCuda.cudaStreamSynchronize(cUstream);
        }
        JCuda.cudaStreamDestroy(cUstream);
        // 释放设备内存
        JCuda.cudaFree(pIn);
        JCudaDriver.cuModuleUnload(module);
    }
}
