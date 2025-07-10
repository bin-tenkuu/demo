package demo.jcuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStream_t;
import org.intellij.lang.annotations.MagicConstant;

import java.io.File;
import java.util.HashMap;

/**
 * @author bin
 * @since 2025/07/09
 */
@SuppressWarnings("unused")
public class JCudaUtil {
    private static final HashMap<String, CUmodule> moduleCache = new HashMap<>();

    // region device

    /**
     * 获取当前CUDA设备的数量
     */
    public static int getCudaDeviceCount() {
        int[] count = new int[1];
        JCuda.cudaGetDeviceCount(count);
        return count[0];
    }

    public static cudaDeviceProp getCudaDeviceProperties(int deviceId) {
        cudaDeviceProp props = new cudaDeviceProp();
        JCuda.cudaGetDeviceProperties(props, deviceId);
        return props;
    }

    // endregion

    // region module function

    public static CUmodule loadModule(File ptxFile) {
        return loadModule(ptxFile.getAbsolutePath());
    }

    public static CUmodule loadModule(String ptxFile) {
        // 检查缓存
        if (moduleCache.containsKey(ptxFile)) {
            return moduleCache.get(ptxFile);
        }

        // 加载PTX模块
        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, ptxFile);
        moduleCache.put(ptxFile, module);
        return module;
    }

    public static CUfunction getFunction(CUmodule module, String functionName) {
        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, functionName);
        return function;
    }

    public static void unloadModuleAll() {
        for (CUmodule module : moduleCache.values()) {
            JCudaDriver.cuModuleUnload(module);
        }
        moduleCache.clear();
    }

    // endregion

    // region malloc

    public static long calcByteSize(int size, @MagicConstant(valuesFromClass = Sizeof.class) int byteSize) {
        return (long) size * byteSize;
    }

    public static CUdeviceptr malloc(long byteSize) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, byteSize);
        return ptr;
    }

    public static CUdeviceptr malloc(int size, @MagicConstant(valuesFromClass = Sizeof.class) int byteSize) {
        return malloc(calcByteSize(size, byteSize));
    }

    public static CUdeviceptr mallocFromArray(Object arr) {
        CUdeviceptr ptr = new CUdeviceptr();
        long byteSize;
        Pointer src;
        switch (arr) {
            case byte[] bs -> {
                byteSize = calcByteSize(bs.length, Sizeof.BYTE);
                src = Pointer.to(bs);
            }
            case char[] cs -> {
                byteSize = calcByteSize(cs.length, Sizeof.CHAR);
                src = Pointer.to(cs);
            }
            case short[] ss -> {
                byteSize = calcByteSize(ss.length, Sizeof.SHORT);
                src = Pointer.to(ss);
            }
            case int[] is -> {
                byteSize = calcByteSize(is.length, Sizeof.INT);
                src = Pointer.to(is);
            }
            case float[] fs -> {
                byteSize = calcByteSize(fs.length, Sizeof.FLOAT);
                src = Pointer.to(fs);
            }
            case long[] ls -> {
                byteSize = calcByteSize(ls.length, Sizeof.LONG);
                src = Pointer.to(ls);
            }
            case double[] ds -> {
                byteSize = calcByteSize(ds.length, Sizeof.DOUBLE);
                src = Pointer.to(ds);
            }
            case null, default -> throw new IllegalStateException("Unexpected value: " + arr);
        }
        JCudaDriver.cuMemAlloc(ptr, byteSize);
        JCuda.cudaMemcpy(ptr, src, byteSize, cudaMemcpyKind.cudaMemcpyHostToDevice);
        return ptr;
    }

    // endregion

    // region stream

    public static cudaStream_t createStream() {
        cudaStream_t stream = new cudaStream_t();
        JCuda.cudaStreamCreate(stream);
        return stream;
    }

    // endregion

    public static int calculateGridSize(int blockSize, int totalSize) {
        return (totalSize + blockSize - 1) / blockSize; // 向上取整
    }

}
