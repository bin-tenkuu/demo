package demo.jcuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStream_t;
import lombok.val;

/**
 * {@code nvcc -ptx nextrand3.cu -o nextrand3.ptx}
 * @author bin
 * @since 2025/07/02
 */
public class H5MotaRandomTest {
    private static final int MAX_INT = 2147483647;
    private static final int INT_RAND = MAX_INT / 3; // 7_1582_7882
    private static final int INT_DIV = MAX_INT / 5; // 20%
    private static final int INT_RATIO = INT_RAND / 100; // 20%

    public static void main(String[] args) {
        val start = System.currentTimeMillis();
        val seeds = new int[1 << 20]; // 1M seeds
        for (var i = 0; i < seeds.length; i++) {
            seeds[i] = i + 1;
        }
        testRandCuda(seeds);
        System.out.printf("Execution time: %d ms%n", System.currentTimeMillis() - start);
    }

    private static void testRand(int seed) {
        int rand = seed;
        int count = 0;
        for (int j = 0; j < INT_RAND; j++) {
            rand = nextRand3(rand);
            count += rand < INT_DIV ? 1 : 0;
        }
        if (count > 143168491) {
            System.out.printf("seed: %10d, ratio: %9d,%3d%n", seed, count, count / INT_RATIO);
        }
    }

    private static void testRandCuda(int[] seeds) {
        var start = System.currentTimeMillis();
        // 创建CUDA上下文
        JCuda.setExceptionsEnabled(true);
        JCuda.cudaSetDevice(0);
        // 编译核函数
        val module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, "test/src/main/java/demo/jcuda/nextrand3.ptx");
        // 获取核函数句柄
        val nextRand3Kernel = new CUfunction();
        JCudaDriver.cuModuleGetFunction(nextRand3Kernel, module, "nextRand3Kernel");

        val size = seeds.length;
        val times = 1 << 20;// INT_RAND; // 1M次随机数生成
        val counts = new int[size];
        val cudaSize = (long) size * Sizeof.INT;
        val ptrSeeds = new CUdeviceptr();
        val ptrCounts = new CUdeviceptr();
        // 创建CUDA流
        val cUstream = new cudaStream_t();
        JCuda.cudaStreamCreate(cUstream);
        JCuda.cudaMalloc(ptrSeeds, cudaSize);
        JCuda.cudaMalloc(ptrCounts, cudaSize);
        JCuda.cudaMemcpyAsync(ptrSeeds, Pointer.to(seeds), cudaSize,
                cudaMemcpyKind.cudaMemcpyHostToDevice, cUstream);
        // 设置内核参数
        val kernelParameters = Pointer.to(
                Pointer.to(ptrSeeds),
                Pointer.to(ptrCounts),
                Pointer.to(new int[]{times}),
                Pointer.to(new int[]{INT_DIV}),
                Pointer.to(new int[]{size})
        );
        // 计算网格和块大小
        int blockSize = 1024; // 每个块1024个线程
        int gridSize = (size + blockSize - 1) / blockSize; // 足够的块覆盖所有元素
        // 启动内核
        JCudaDriver.cuLaunchKernel(nextRand3Kernel,
                gridSize, 1, 1,     // 网格维度
                blockSize, 1, 1,    // 块维度
                0, new CUstream(cUstream),            // 共享内存和流
                kernelParameters, null // 内核参数
        );
        // 将结果复制回主机
        JCuda.cudaMemcpyAsync(Pointer.to(counts), ptrCounts, cudaSize,
                cudaMemcpyKind.cudaMemcpyDeviceToHost, cUstream);
        JCuda.cudaLaunchHostFunc(cUstream, time -> {
            System.out.println("Kernel execution completed in " + (System.currentTimeMillis() - (long) time) + " ms");
        }, start);
        JCuda.cudaStreamSynchronize(cUstream);// 等待内核完成
        JCuda.cudaStreamDestroy(cUstream);

        // 释放设备内存
        JCuda.cudaFree(ptrSeeds);
        JCuda.cudaFree(ptrCounts);
        JCudaDriver.cuModuleUnload(module);

        int minIdx = 0, maxIdx = 0;
        int minCount = counts[minIdx], maxCount = counts[maxIdx];
        for (var i = 0; i < seeds.length; i++) {
            val count = counts[i];
            if (count < minCount) {
                minIdx = i;
                minCount = count;
            } else if (count > maxCount) {
                maxIdx = i;
                maxCount = count;
            }
        }
        System.out.printf("Min: seed: %10d, ratio: %9d,%3s%n", seeds[minIdx], counts[minIdx],
                counts[minIdx] / (times / 100.0));
        System.out.printf("Max: seed: %10d, ratio: %9d,%3s%n", seeds[maxIdx], counts[maxIdx],
                counts[maxIdx] / (times / 100.0));
    }

    private static final int DIVISOR = 127773;
    private static final int MULT1 = 16807;
    private static final int MULT2 = 2836;

    private static int nextRand(int rand) {
        rand = rand % DIVISOR * MULT1 - (rand / DIVISOR) * MULT2;
        rand += rand >> 31;
        return rand;
    }

    private static int nextRand3(int rand) {
        // 1
        int q = rand / DIVISOR;
        int r = rand - q * DIVISOR;
        rand = r * MULT1 - q * MULT2;
        rand += rand >> 31 & MAX_INT;
        // 2
        q = rand / DIVISOR;
        r = rand - q * DIVISOR;
        rand = r * MULT1 - q * MULT2;
        rand += rand >> 31 & MAX_INT;
        // 3
        q = rand / DIVISOR;
        r = rand - q * DIVISOR;
        rand = r * MULT1 - q * MULT2;
        rand += rand >> 31 & MAX_INT;
        return rand;
    }
}
/*
core.utils.__init_seed = () => {
    core.setFlag("__seed__", 0);
    core.setFlag("__rand__", 0)
}
core.utils.__init_seed = () => {
    utils.prototype.__init_seed.call(core.utils);
    console.log(`__init_seed ${core.getFlag("__seed__")}`);
}
core.utils.__next_rand = _rand => {
    let n = utils.prototype.__next_rand.call(core.utils, _rand);
    console.log(`__next_rand ${_rand} => ${n}`);
    return n;
}
*/
