extern "C" __global__ void nextRand3Kernel(int *seeds, int *counts, int times, int compare, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int rand = seeds[idx];
        for (int i = 0; i < times; i++) {

            // 第一次计算
            int q = rand / 127773;
            int r = rand - q * 127773;
            rand = r * 16807 - q * 2836;
            rand += (rand >> 31) & 0x7FFFFFFF;

            // 第二次计算
            q = rand / 127773;
            r = rand - q * 127773;
            rand = r * 16807 - q * 2836;
            rand += (rand >> 31) & 0x7FFFFFFF;

            // 第三次计算
            q = rand / 127773;
            r = rand - q * 127773;
            rand = r * 16807 - q * 2836;
            rand += (rand >> 31) & 0x7FFFFFFF;

            counts[idx] += (rand < compare) ? 1 : 0;

        }
        seeds[idx] = rand;

    }
}
