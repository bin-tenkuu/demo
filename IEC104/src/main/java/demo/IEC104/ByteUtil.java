package demo.IEC104;

import lombok.val;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * @author bin
 * @since 2024/06/13
 */
public interface ByteUtil {
    VarHandle INT_ARRAY = MethodHandles
            .byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();
    VarHandle SHORT_ARRAY = MethodHandles
            .byteArrayViewVarHandle(short[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();
    VarHandle FLOAT_ARRAY = MethodHandles
            .byteArrayViewVarHandle(float[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();

    /**
     * 从字节数组中字节数组索引位置向后获取 int
     */
    static int getInt(byte[] bytes, int index) {
        return (int) INT_ARRAY.get(bytes, index);
    }

    /**
     * 从字节数组中字节数组索引位置向后设置 int
     */
    static void setInt(byte[] bytes, int index, int value) {
        INT_ARRAY.set(bytes, index, value);
    }

    /**
     * 从字节数组中字节数组索引位置向后获取 short
     */
    static short getShort(byte[] bytes, int index) {
        return (short) SHORT_ARRAY.get(bytes, index);
    }

    /**
     * 从字节数组中字节数组索引位置向后设置 short
     */
    static void setShort(byte[] bytes, int index, short value) {
        SHORT_ARRAY.set(bytes, index, value);
    }

    /**
     * 从字节数组中字节数组索引位置向后获取 float
     */
    static float getFloat(byte[] bytes, int index) {
        return (float) FLOAT_ARRAY.get(bytes, index);
    }

    /**
     * 从字节数组中字节数组索引位置向后设置 float
     */
    static void setFloat(byte[] bytes, int index, float value) {
        FLOAT_ARRAY.set(bytes, index, value);
    }

    /**
     * @param index 0-7
     */
    static byte setBit(byte b, int index, boolean value) {
        if (value) {
            return (byte) (b | 1 << index);
        } else {
            return (byte) (b & ~(1 << index));
        }
    }

    /**
     * @param index 0-7
     */
    static boolean getBit(byte b, int index) {
        return (b & 1 << index) != 0;
    }

    static void toString(StringBuilder sb, byte[] array, int offset, int length) {
        for (; length > 0; length--, offset++) {
            byte b = array[offset];
            sb.append(toString(b)).append(' ');
        }
    }

    static void toString(StringBuilder sb, byte[] array) {
        for (byte b : array) {
            sb.append(toString(b)).append(' ');
        }
    }

    static String toString(byte[] array) {
        val sb = new StringBuilder(array.length * 3);
        toString(sb, array);
        return sb.toString();
    }

    static String toString(byte b) {
        val i = b & 0xFF;
        return (i < 16 ? "0" : "") + Integer.toHexString(i);
    }

    static byte[] fromString(String string) {
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
