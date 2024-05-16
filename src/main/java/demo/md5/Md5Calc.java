package demo.md5;

import cn.hutool.crypto.digest.DigestAlgorithm;
import cn.hutool.crypto.digest.Digester;
import lombok.val;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
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
            0, 0// 2105668443402L
    };
    private static final int length = 4;
    public static final List<String> msgs = new LinkedList<>();
    public static boolean flag = false;
    private final int n;
    private final int part;

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
        MD5.digest(before, after);
        while (!flag) {
            add(part);
            toArray(before, a, b);
            MD5.digest(before, after);
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

    public static void main() {
        final byte[] in = new byte[]{
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 92, -97, 88
        };
        printf(in);
        byte[] out = new Digester(DigestAlgorithm.MD5).digest(in);
        printf(out);
        try {
            val md5 = MessageDigest.getInstance("MD5");
            printf(md5.digest(in));
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
        MD5.digest(in, out);
        printf(out);
    }

    private static void printf(byte[] bs) {
        for (byte b : bs) {
            // hex
            System.out.printf("%02x", b);

        }
        System.out.println();
    }
}
