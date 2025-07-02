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
        JCuda.initialize();
        // 启用异常捕获（重要！）
        JCuda.setExceptionsEnabled(true);

        try {
            // 步骤1：初始化CUDA
            JCuda.cudaSetDevice(0);  // 使用第一个GPU

            // 步骤2：获取设备信息
            cudaDeviceProp props = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(props, 0);

            // 打印设备信息
            String deviceName = new String(props.name).trim();
            System.out.println("[SUCCESS] 检测到CUDA设备:");
            System.out.println("设备名称: " + deviceName);
            System.out.println("计算能力: " + props.major + "." + props.minor);
            System.out.println("显存大小: " + (props.totalGlobalMem >> 20) + " MB");

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
