#include<stdio.h>
// 旋转左移函数 (匹配Java实现)
__device__ int rotateLeft(int i, int distance) {
    return (i << distance) | ((unsigned int)i >> (32 - distance));
}

// 辅助函数 FF (匹配Java实现)
__device__ int FF(int a, int b, int c, int d, int x, int s, int ac) {
    return rotateLeft(a + (((b & c) | ((~b) & d)) + x + ac), s) + b;
}

// 辅助函数 GG (匹配Java实现)
__device__ int GG(int a, int b, int c, int d, int x, int s, int ac) {
    return rotateLeft(a + (((b & d) | (c & (~d))) + x + ac), s) + b;
}

// 辅助函数 HH (匹配Java实现)
__device__ int HH(int a, int b, int c, int d, int x, int s, int ac) {
    return rotateLeft(a + (((b ^ c) ^ d) + x + ac), s) + b;
}

// 辅助函数 II (匹配Java实现)
__device__ int II(int a, int b, int c, int d, int x, int s, int ac) {
    return rotateLeft(a + ((c ^ (b | (~d))) + x + ac), s) + b;
}

__device__ void calc(const int x0, const int x1, const int x2, const int x3, int* out) {
    int a = 0x67452301;
    int b = 0xefcdab89;
    int c = 0x98badcfe;
    int d = 0x10325476;

    a = FF(a, b, c, d, x0, 7, 0xd76aa478);
    d = FF(d, a, b, c, x1, 12, 0xe8c7b756);
    c = FF(c, d, a, b, x2, 17, 0x242070db);
    b = FF(b, c, d, a, x3, 22, 0xc1bdceee);
    a = FF(a, b, c, d, 128, 7, 0xf57c0faf);
    d = FF(d, a, b, c, 0, 12, 0x4787c62a);
    c = FF(c, d, a, b, 0, 17, 0xa8304613);
    b = FF(b, c, d, a, 0, 22, 0xfd469501);
    a = FF(a, b, c, d, 0, 7, 0x698098d8);
    d = FF(d, a, b, c, 0, 12, 0x8b44f7af);
    c = FF(c, d, a, b, 0, 17, 0xffff5bb1);
    b = FF(b, c, d, a, 0, 22, 0x895cd7be);
    a = FF(a, b, c, d, 0, 7, 0x6b901122);
    d = FF(d, a, b, c, 0, 12, 0xfd987193);
    c = FF(c, d, a, b, 128, 17, 0xa679438e);
    b = FF(b, c, d, a, 0, 22, 0x49b40821);

    a = GG(a, b, c, d, x1, 5, 0xf61e2562);
    d = GG(d, a, b, c, 0, 9, 0xc040b340);
    c = GG(c, d, a, b, 0, 14, 0x265e5a51);
    b = GG(b, c, d, a, x0, 20, 0xe9b6c7aa);
    a = GG(a, b, c, d, 0, 5, 0xd62f105d);
    d = GG(d, a, b, c, 0, 9, 0x2441453);
    c = GG(c, d, a, b, 0, 14, 0xd8a1e681);
    b = GG(b, c, d, a, 128, 20, 0xe7d3fbc8);
    a = GG(a, b, c, d, 0, 5, 0x21e1cde6);
    d = GG(d, a, b, c, 128, 9, 0xc33707d6);
    c = GG(c, d, a, b, x3, 14, 0xf4d50d87);
    b = GG(b, c, d, a, 0, 20, 0x455a14ed);
    a = GG(a, b, c, d, 0, 5, 0xa9e3e905);
    d = GG(d, a, b, c, x2, 9, 0xfcefa3f8);
    c = GG(c, d, a, b, 0, 14, 0x676f02d9);
    b = GG(b, c, d, a, 0, 20, 0x8d2a4c8a);

    a = HH(a, b, c, d, 0, 4, 0xfffa3942);
    d = HH(d, a, b, c, 0, 11, 0x8771f681);
    c = HH(c, d, a, b, 0, 16, 0x6d9d6122);
    b = HH(b, c, d, a, 128, 23, 0xfde5380c);
    a = HH(a, b, c, d, x1, 4, 0xa4beea44);
    d = HH(d, a, b, c, 128, 11, 0x4bdecfa9);
    c = HH(c, d, a, b, 0, 16, 0xf6bb4b60);
    b = HH(b, c, d, a, 0, 23, 0xbebfbc70);
    a = HH(a, b, c, d, 0, 4, 0x289b7ec6);
    d = HH(d, a, b, c, x0, 11, 0xeaa127fa);
    c = HH(c, d, a, b, x3, 16, 0xd4ef3085);
    b = HH(b, c, d, a, 0, 23, 0x4881d05);
    a = HH(a, b, c, d, 0, 4, 0xd9d4d039);
    d = HH(d, a, b, c, 0, 11, 0xe6db99e5);
    c = HH(c, d, a, b, 0, 16, 0x1fa27cf8);
    b = HH(b, c, d, a, x2, 23, 0xc4ac5665);

    a = II(a, b, c, d, x0, 6, 0xf4292244);
    d = II(d, a, b, c, 0, 10, 0x432aff97);
    c = II(c, d, a, b, 128, 15, 0xab9423a7);
    b = II(b, c, d, a, 0, 21, 0xfc93a039);
    a = II(a, b, c, d, 0, 6, 0x655b59c3);
    d = II(d, a, b, c, x3, 10, 0x8f0ccc92);
    c = II(c, d, a, b, 0, 15, 0xffeff47d);
    b = II(b, c, d, a, x1, 21, 0x85845dd1);
    a = II(a, b, c, d, 0, 6, 0x6fa87e4f);
    d = II(d, a, b, c, 0, 10, 0xfe2ce6e0);
    c = II(c, d, a, b, 0, 15, 0xa3014314);
    b = II(b, c, d, a, 0, 21, 0x4e0811a1);
    a = II(a, b, c, d, 128, 6, 0xf7537e82);
    d = II(d, a, b, c, 0, 10, 0xbd3af235);
    c = II(c, d, a, b, x2, 15, 0x2ad7d2bb);
    b = II(b, c, d, a, 0, 21, 0xeb86d391);

    out[0] = 0x67452301 + a;
    out[1] = 0xefcdab89 + b;
    out[2] = 0x98badcfe + c;
    out[3] = 0x10325476 + d;
}

extern "C" __global__
void md5(
    const int* before  // 输入数组 (16字节 = 4个int)
) {
    const int x0 = before[0];
    const int x1 = before[1];
    const int x2 = blockIdx.x + before[2];
    int x3;
    int out[4] = {0, 0, 0, 0};

    for (int i = 0; i < 4194304; i++) {
        x3 = threadIdx.x + i * blockDim.x;

        calc(x0, x1, x2, x3, out);

        if (
            x0 == out[0]
         && x1 == out[1]
         && x2 == out[2]
//          x3 == 0
        ) {
            printf("%d,%d,%d,%d\t\t%d,%d,%d,%d\n", x0, x1, x2, x3, out[0], out[1], out[2], out[3]);
        }
    }

}
