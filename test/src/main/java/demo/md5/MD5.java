
package demo.md5;

import lombok.val;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

public final class MD5 {
    private static final VarHandle INT_ARRAY
            = MethodHandles.byteArrayViewVarHandle(int[].class,
            ByteOrder.LITTLE_ENDIAN).withInvokeExactBehavior();

    public static void main(String[] args) {
        int[] in = new int[]{
                0,
                0,
                0,
                0,
        };
        val out = digest(in);
        System.out.printf("%d,%d,%d,%d\t%d,%d,%d,%d\n", in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3]);
    }

    public static byte[] digest(byte[] bs) {
        val state = new int[]{
                (int) INT_ARRAY.get(bs, 0),
                (int) INT_ARRAY.get(bs, 4),
                (int) INT_ARRAY.get(bs, 8),
                (int) INT_ARRAY.get(bs, 12),
        };

        implCompress(state);
        byte[] out = new byte[16];
        INT_ARRAY.set(out, 0, state[0]);
        INT_ARRAY.set(out, 4, state[1]);
        INT_ARRAY.set(out, 8, state[2]);
        INT_ARRAY.set(out, 12, state[3]);
        return out;
    }

    public static int[] digest(int[] is) {
        val state = is.clone();
        implCompress(state);
        return state;
    }

    public static int rotateLeft(int i, int distance) {
        return (i << distance) | (i >>> -distance);
    }

    private static int FF(int a, int b, int c, int d, int x, int s, int ac) {
        return rotateLeft(a + (((b & c) | ((~b) & d)) + x + ac), s) + b;
    }

    private static int GG(int a, int b, int c, int d, int x, int s, int ac) {
        return rotateLeft(a + (((b & d) | (c & (~d))) + x + ac), s) + b;
    }

    private static int HH(int a, int b, int c, int d, int x, int s, int ac) {
        return rotateLeft(a + (((b ^ c) ^ d) + x + ac), s) + b;
    }

    private static int II(int a, int b, int c, int d, int x, int s, int ac) {
        return rotateLeft(a + ((c ^ (b | (~d))) + x + ac), s) + b;
    }

    private static void implCompress(int[] state) {
        int a = 0x67452301;
        int b = 0xefcdab89;
        int c = 0x98badcfe;
        int d = 0x10325476;

        int x0 = state[0];
        int x1 = state[1];
        int x2 = state[2];
        int x3 = state[3];

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

        state[0] = 0x67452301 + a;
        state[1] = 0xefcdab89 + b;
        state[2] = 0x98badcfe + c;
        state[3] = 0x10325476 + d;
    }

}
