package demo.IEC104;

import lombok.val;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * @author bin
 * @since 2024/06/13
 */
public class ByteUtil {
    public static final VarHandle INT_ARRAY = MethodHandles
            .byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();
    public static final VarHandle SHORT_ARRAY = MethodHandles
            .byteArrayViewVarHandle(short[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();
    public static final VarHandle FLOAT_ARRAY = MethodHandles
            .byteArrayViewVarHandle(float[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();

    /**
     * 从字节数组中字节数组索引位置向后获取 int
     */
    public static int getInt(byte[] bytes, int index) {
        return (int) INT_ARRAY.get(bytes, index);
    }

    /**
     * 从字节数组中字节数组索引位置向后设置 int
     */
    public static void setInt(byte[] bytes, int index, int value) {
        INT_ARRAY.set(bytes, index, value);
    }

    /**
     * 从字节数组中字节数组索引位置向后获取 short
     */
    public static short getShort(byte[] bytes, int index) {
        return (short) SHORT_ARRAY.get(bytes, index);
    }

    /**
     * 从字节数组中字节数组索引位置向后设置 short
     */
    public static void setShort(byte[] bytes, int index, short value) {
        SHORT_ARRAY.set(bytes, index, value);
    }

    /**
     * 从字节数组中字节数组索引位置向后获取 float
     */
    public static float getFloat(byte[] bytes, int index) {
        return (float) FLOAT_ARRAY.get(bytes, index);
    }

    /**
     * 从字节数组中字节数组索引位置向后设置 float
     */
    public static void setFloat(byte[] bytes, int index, float value) {
        FLOAT_ARRAY.set(bytes, index, value);
    }

    /**
     * @param index 0-7
     */
    public static byte setBit(byte b, int index, boolean value) {
        if (value) {
            return (byte) (b | 1 << index);
        } else {
            return (byte) (b & ~(1 << index));
        }
    }

    /**
     * @param index 0-7
     */
    public static boolean getBit(byte b, int index) {
        return (b & 1 << index) != 0;
    }

    public static String toString(byte[] array) {
        val builder = new StringBuilder(array.length * 3);
        for (byte b : array) {
            builder.append(toString(b)).append(' ');
        }
        return builder.toString();
    }

    public static String toString(byte b) {
        return (b >= 0 && b < 16 ? "0" : "") + Integer.toHexString(b & 0xFF);
    }

    public static byte[] fromString(String string) {
        val length = string.length();
        val bytes = new byte[length / 2];
        int bi = 0;
        for (int i = 0; i < length; i++) {
            val c = string.charAt(i);
            if (Character.isDigit(c) || Character.isLetter(c)) {
                bytes[bi] = (byte) Integer.parseInt(string.substring(i, i + 2), 16);
                bi++;
                i++;
            }
        }
        val copy = new byte[bi];
        System.arraycopy(bytes, 0, copy, 0, bi);

        return copy;
    }

}
