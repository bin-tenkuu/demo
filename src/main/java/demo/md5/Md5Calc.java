package demo.md5;

import lombok.val;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * @author bin
 * @since 2024/05/15
 */
@SuppressWarnings("preview")
public final class Md5Calc extends Thread {
    private static final long[] init = {
            0, 17094188568L// 348556822170L
    };
    private static final int length = 4;
    public static final List<String> msgs = new LinkedList<>();
    public static boolean flag = false;
    private final int n;
    private final int part;
    private final MD5 md5 = new MD5();

    public long a = init[0];
    public long b = init[1];

    public Md5Calc(int n, int part) {
        this.n = n;
        this.part = part;
    }

    @Override
    public void run() {
        byte[] before = new byte[16];
        byte[] after = new byte[16];
        add(n);
        toArray(before, a, b);
        md5.digest(before, after);
        while (!flag) {
            add(part);
            toArray(before, a, b);
            md5.digest(before, after);
            compare(before, after);
        }
    }

    private void add(int n) {
        val r = b = b + n;
        if (r >= 0 && r < n) {
            a++;
        }
    }

    private static final VarHandle Long_ARRAY = MethodHandles.byteArrayViewVarHandle(long[].class,
            ByteOrder.BIG_ENDIAN).withInvokeExactBehavior();

    private static void toArray(byte[] bs, long a, long b) {
        Long_ARRAY.set(bs, 0, a);
        Long_ARRAY.set(bs, 8, b);
    }

    private static void compare(byte[] before, byte[] after) {
        for (int i = 0; i < length; i++) {
            if (before[i] != after[i]) {
                return;
            }
        }
        val msg = STR."same: raw:\{Arrays.toString(before)} md5:\{Arrays.toString(after)}\n";
        msgs.add(msg);
        System.out.print(msg);
    }

}
