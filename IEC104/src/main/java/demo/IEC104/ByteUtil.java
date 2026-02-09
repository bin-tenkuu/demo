package demo.IEC104;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * @author bin
 * @since 2024/06/13
 */
public interface ByteUtil {
    VarHandle SHORT_ARRAY = MethodHandles
            .byteArrayViewVarHandle(short[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();
    VarHandle INT_ARRAY = MethodHandles
            .byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();
    VarHandle LONG_ARRAY = MethodHandles
            .byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();
    VarHandle FLOAT_ARRAY = MethodHandles
            .byteArrayViewVarHandle(float[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();
    VarHandle DOUBLE_ARRAY = MethodHandles
            .byteArrayViewVarHandle(double[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();

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
     * 从字节数组中字节数组索引位置向后获取 long
     */
    static long getLong(byte[] bytes, int index) {
        return (long) LONG_ARRAY.get(bytes, index);
    }

    /**
     * 从字节数组中字节数组索引位置向后设置 long
     */
    static void setLong(byte[] bytes, int index, long value) {
        LONG_ARRAY.set(bytes, index, value);
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
     * 从字节数组中字节数组索引位置向后获取 double
     */
    static double getDouble(byte[] bytes, int index) {
        return (double) DOUBLE_ARRAY.get(bytes, index);
    }

    /**
     * 从字节数组中字节数组索引位置向后设置 double
     */
    static void setDouble(byte[] bytes, int index, double value) {
        DOUBLE_ARRAY.set(bytes, index, value);
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

    String HEX = "0123456789ABCDEF";

    static void toString(StringBuilder sb, byte[] array, int offset, int length) {
        for (; length > 0; length--, offset++) {
            byte b = array[offset];
            toString(sb, b);
            sb.append(' ');
        }
    }

    static void toString(StringBuilder sb, byte[] array) {
        for (byte b : array) {
            toString(sb, b);
            sb.append(' ');
        }
    }

    static void toString(StringBuilder sb, byte b) {
        sb.append(HEX.charAt(b >> 4 & 0xF));
        sb.append(HEX.charAt(b & 0xF));
    }

    static String toString(byte[] array) {
        var sb = new StringBuilder(array.length * 3);
        toString(sb, array);
        return sb.toString();
    }

    static String toString(byte b) {
        return new String(new char[]{
                HEX.charAt(b >> 4 & 0xF),
                HEX.charAt(b & 0xF)
        });
    }

    static byte[] fromString(String hexString) {
        Objects.requireNonNull(hexString, "Input hexString cannot be null");
        int length = hexString.length();
        var hexChars = new char[length];
        var charSize = 0;
        for (var i = 0; i < length; i++) {
            var c = hexString.charAt(i);
            if ((c >= '0' && c <= '9') ||
                    (c >= 'a' && c <= 'f') ||
                    (c >= 'A' && c <= 'F')) {
                hexChars[charSize] = c;
                charSize++;
            }
        }
        if (charSize % 2 != 0) {
            throw new IllegalArgumentException("Invalid hex string: odd number of hex digits");
        }
        var bytes = new byte[charSize / 2];
        for (int i = 0; i < charSize; i += 2) {
            var high = Character.digit(hexChars[i], 16);
            var low = Character.digit(hexChars[i + 1], 16);
            var b = (byte) ((high << 4) + low);
            bytes[i / 2] = b;
        }
        return bytes;
    }

    static void main(String[] args) {
        var bytes = fromString("00 00 00 00 00 e0 90 40");
        System.out.println(getDouble(bytes, 0));
        System.out.println(toString(bytes));
    }
}
