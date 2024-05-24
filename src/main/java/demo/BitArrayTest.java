package demo;

import lombok.val;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * @author bin
 * @since 2024/05/20
 */

public class BitArrayTest {
    public static final VarHandle LONG_ARRAY_B = MethodHandles.byteArrayViewVarHandle(long[].class,
            ByteOrder.BIG_ENDIAN).withInvokeExactBehavior();
    private static final long M = 0x0101010101010101L;

    public static void main() {
        println((byte) 0b10000000);
        println(toBitArray_ge((byte) 0b10000000));
        println(toBitArraySIMD_ge((byte) 0b10000000));
    }

    public static byte[] toBitArray_ge(byte s) {
        val bs = new byte[8];
        bs[7] = (byte) (s >>> 0 & 1);
        bs[6] = (byte) (s >>> 1 & 1);
        bs[5] = (byte) (s >>> 2 & 1);
        bs[4] = (byte) (s >>> 3 & 1);
        bs[3] = (byte) (s >>> 4 & 1);
        bs[2] = (byte) (s >>> 5 & 1);
        bs[1] = (byte) (s >>> 6 & 1);
        bs[0] = (byte) (s >>> 7 & 1);
        return bs;
    }

    public static byte[] toBitArraySIMD_ge(byte x) {
        val m = 0x2040810204081L;
        long y = ((x & 0xfe) * m | x & 0xff) & M;
        val bs = new byte[8];
        LONG_ARRAY_B.set(bs, 0, y);
        return bs;
    }

    private static void println(byte b) {
        System.out.println(Integer.toBinaryString(b).substring(24));
    }

    private static void println(byte[] bs) {
        for (byte b : bs) {
            System.out.print(b == 1 ? 1 : 0);
        }
        System.out.println();
    }
}
