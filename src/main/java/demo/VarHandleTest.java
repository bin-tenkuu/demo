package demo;

import lombok.val;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/07/04
 */
public class VarHandleTest {
    public static final VarHandle LONG_ARRAY_B = MethodHandles
            .byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN)
            .withInvokeExactBehavior();
    private static final byte[] bytes = new byte[16];

    public static void main() {
        LONG_ARRAY_B.set(bytes, 0, 1L);
        System.out.println((long) LONG_ARRAY_B.get(bytes, 0));
        val segment = MemorySegment.ofArray(bytes);
        System.out.println(segment.getAtIndex(ValueLayout.JAVA_LONG, 0));

    }
}
