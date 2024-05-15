
package demo.md5;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

final class MD5 {
    private static final VarHandle INT_ARRAY = MethodHandles.byteArrayViewVarHandle(int[].class,
            ByteOrder.LITTLE_ENDIAN).withInvokeExactBehavior();

    // state of this object
    private final int[] state = new int[4];

    public MD5() {
    }

    private static int FF(int a, int b, int c, int d, int x, int s, int ac) {
        a += ((b & c) | ((~b) & d)) + x + ac;
        return Integer.rotateLeft(a, s) + b;
    }

    private static int GG(int a, int b, int c, int d, int x, int s, int ac) {
        a += ((b & d) | (c & (~d))) + x + ac;
        return Integer.rotateLeft(a, s) + b;
    }

    private static int HH(int a, int b, int c, int d, int x, int s, int ac) {
        a += ((b ^ c) ^ d) + x + ac;
        return Integer.rotateLeft(a, s) + b;
    }

    private static int II(int a, int b, int c, int d, int x, int s, int ac) {
        a += (c ^ (b | (~d))) + x + ac;
        return Integer.rotateLeft(a, s) + b;
    }

    public void digest(byte[] in, byte[] out) {
        implCompress(
                (int) INT_ARRAY.get(in, 0),
                (int) INT_ARRAY.get(in, 4),
                (int) INT_ARRAY.get(in, 8),
                (int) INT_ARRAY.get(in, 12)
        );

        INT_ARRAY.set(out, 0, state[0]);
        INT_ARRAY.set(out, 4, state[1]);
        INT_ARRAY.set(out, 8, state[2]);
        INT_ARRAY.set(out, 12, state[3]);
    }

    private void implCompress(
            final int x0,
            final int x1,
            final int x2,
            final int x3
    ) {
        int a = 0x67452301;
        int b = 0xefcdab89;
        int c = 0x98badcfe;
        int d = 0x10325476;

        /* Round 1 */
        a = FF(a, b, c, d, x0, 7, 0xd76aa478); /* 1 */
        d = FF(d, a, b, c, x1, 12, 0xe8c7b756); /* 2 */
        c = FF(c, d, a, b, x2, 17, 0x242070db); /* 3 */
        b = FF(b, c, d, a, x3, 22, 0xc1bdceee); /* 4 */
        a = FF(a, b, c, d, 128, 7, 0xf57c0faf); /* 5 */
        d = FF(d, a, b, c, 0, 12, 0x4787c62a); /* 6 */
        c = FF(c, d, a, b, 0, 17, 0xa8304613); /* 7 */
        b = FF(b, c, d, a, 0, 22, 0xfd469501); /* 8 */
        a = FF(a, b, c, d, 0, 7, 0x698098d8); /* 9 */
        d = FF(d, a, b, c, 0, 12, 0x8b44f7af); /* 10 */
        c = FF(c, d, a, b, 0, 17, 0xffff5bb1); /* 11 */
        b = FF(b, c, d, a, 0, 22, 0x895cd7be); /* 12 */
        a = FF(a, b, c, d, 0, 7, 0x6b901122); /* 13 */
        d = FF(d, a, b, c, 0, 12, 0xfd987193); /* 14 */
        c = FF(c, d, a, b, 128, 17, 0xa679438e); /* 15 */
        b = FF(b, c, d, a, 0, 22, 0x49b40821); /* 16 */

        /* Round 2 */
        a = GG(a, b, c, d, x1, 5, 0xf61e2562); /* 17 */
        d = GG(d, a, b, c, 0, 9, 0xc040b340); /* 18 */
        c = GG(c, d, a, b, 0, 14, 0x265e5a51); /* 19 */
        b = GG(b, c, d, a, x0, 20, 0xe9b6c7aa); /* 20 */
        a = GG(a, b, c, d, 0, 5, 0xd62f105d); /* 21 */
        d = GG(d, a, b, c, 0, 9, 0x2441453); /* 22 */
        c = GG(c, d, a, b, 0, 14, 0xd8a1e681); /* 23 */
        b = GG(b, c, d, a, 128, 20, 0xe7d3fbc8); /* 24 */
        a = GG(a, b, c, d, 0, 5, 0x21e1cde6); /* 25 */
        d = GG(d, a, b, c, 128, 9, 0xc33707d6); /* 26 */
        c = GG(c, d, a, b, x3, 14, 0xf4d50d87); /* 27 */
        b = GG(b, c, d, a, 0, 20, 0x455a14ed); /* 28 */
        a = GG(a, b, c, d, 0, 5, 0xa9e3e905); /* 29 */
        d = GG(d, a, b, c, x2, 9, 0xfcefa3f8); /* 30 */
        c = GG(c, d, a, b, 0, 14, 0x676f02d9); /* 31 */
        b = GG(b, c, d, a, 0, 20, 0x8d2a4c8a); /* 32 */

        /* Round 3 */
        a = HH(a, b, c, d, 0, 4, 0xfffa3942); /* 33 */
        d = HH(d, a, b, c, 0, 11, 0x8771f681); /* 34 */
        c = HH(c, d, a, b, 0, 16, 0x6d9d6122); /* 35 */
        b = HH(b, c, d, a, 128, 23, 0xfde5380c); /* 36 */
        a = HH(a, b, c, d, x1, 4, 0xa4beea44); /* 37 */
        d = HH(d, a, b, c, 128, 11, 0x4bdecfa9); /* 38 */
        c = HH(c, d, a, b, 0, 16, 0xf6bb4b60); /* 39 */
        b = HH(b, c, d, a, 0, 23, 0xbebfbc70); /* 40 */
        a = HH(a, b, c, d, 0, 4, 0x289b7ec6); /* 41 */
        d = HH(d, a, b, c, x0, 11, 0xeaa127fa); /* 42 */
        c = HH(c, d, a, b, x3, 16, 0xd4ef3085); /* 43 */
        b = HH(b, c, d, a, 0, 23, 0x4881d05); /* 44 */
        a = HH(a, b, c, d, 0, 4, 0xd9d4d039); /* 45 */
        d = HH(d, a, b, c, 0, 11, 0xe6db99e5); /* 46 */
        c = HH(c, d, a, b, 0, 16, 0x1fa27cf8); /* 47 */
        b = HH(b, c, d, a, x2, 23, 0xc4ac5665); /* 48 */

        /* Round 4 */
        a = II(a, b, c, d, x0, 6, 0xf4292244); /* 49 */
        d = II(d, a, b, c, 0, 10, 0x432aff97); /* 50 */
        c = II(c, d, a, b, 128, 15, 0xab9423a7); /* 51 */
        b = II(b, c, d, a, 0, 21, 0xfc93a039); /* 52 */
        a = II(a, b, c, d, 0, 6, 0x655b59c3); /* 53 */
        d = II(d, a, b, c, x3, 10, 0x8f0ccc92); /* 54 */
        c = II(c, d, a, b, 0, 15, 0xffeff47d); /* 55 */
        b = II(b, c, d, a, x1, 21, 0x85845dd1); /* 56 */
        a = II(a, b, c, d, 0, 6, 0x6fa87e4f); /* 57 */
        d = II(d, a, b, c, 0, 10, 0xfe2ce6e0); /* 58 */
        c = II(c, d, a, b, 0, 15, 0xa3014314); /* 59 */
        b = II(b, c, d, a, 0, 21, 0x4e0811a1); /* 60 */
        a = II(a, b, c, d, 128, 6, 0xf7537e82); /* 61 */
        d = II(d, a, b, c, 0, 10, 0xbd3af235); /* 62 */
        c = II(c, d, a, b, x2, 15, 0x2ad7d2bb); /* 63 */
        b = II(b, c, d, a, 0, 21, 0xeb86d391); /* 64 */

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }

}
