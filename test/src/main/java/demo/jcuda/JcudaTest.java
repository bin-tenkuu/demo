package demo.jcuda;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

/**
 * @author bin
 * @since 2025/07/02
 */
public class JcudaTest {
    public static void main(String[] args) {
        // 启用异常捕获（重要！）
        JCuda.setExceptionsEnabled(true);

        try {
            // 步骤1：初始化CUDA
            JCuda.cudaSetDevice(0);  // 使用第一个GPU

            // 步骤2：获取设备信息
            cudaDeviceProp props = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(props, 0);

            // 打印设备信息
            System.out.println("[SUCCESS] 检测到CUDA设备:");
            System.out.printf("设备名称: %s\n", new String(props.name).trim());
            System.out.printf("计算能力: %d.%d\n", props.major, props.minor);
            System.out.printf("显存大小: %.2f GB\n", (props.totalGlobalMem >> 20) / 1024.0);
            System.out.printf("多处理器数量: %d\n", props.multiProcessorCount);
            System.out.printf("最大线程数: %d\n", props.maxThreadsPerBlock);
            System.out.printf("最大线程维度: (%d, %d, %d)\n",
                    props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
            System.out.printf("最大块维度: (%d, %d, %d)\n",
                    props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);

            // 步骤3：执行简单内存操作（核心功能测试）
            Pointer devicePointer = new Pointer();
            JCuda.cudaMalloc(devicePointer, 4);  // 分配4字节内存
            JCuda.cudaFree(devicePointer);        // 释放内存
            System.out.println("CUDA内存操作测试通过！");

        } catch (Exception e) {
            System.err.println("[ERROR] JCUDA运行失败:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
