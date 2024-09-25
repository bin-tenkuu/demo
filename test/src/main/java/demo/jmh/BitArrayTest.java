package demo.jmh;

import lombok.val;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.concurrent.TimeUnit;

/**
 * @author bin
 * @since 2024/05/20
 */

@BenchmarkMode(Mode.AverageTime)
@State(Scope.Benchmark)
@Fork(1)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 3)
public class BitArrayTest {
    public static final VarHandle LONG_ARRAY_B = MethodHandles
            .byteArrayViewVarHandle(long[].class, ByteOrder.BIG_ENDIAN)
            .withInvokeExactBehavior();
    private static final long M = 0x0101010101010101L;

    public static void main() {
        Options opt = new OptionsBuilder()
                .include(BitArrayTest.class.getSimpleName())
                // .result("result.json")
                // .resultFormat(ResultFormatType.JSON)
                .build();
        try {
            new Runner(opt).run();
        } catch (RunnerException e) {
            throw new RuntimeException(e);
        }
    }

    @Benchmark
    public void normal() {
        val bs = new byte[8];
        for (int n = 0; n < 1000000000; n++) {
            for (int i = 0; i < 255; i++) {
                toBitArray_ge(bs, (byte) i);
            }
        }
    }

    @Benchmark
    public void simd() {
        val bs = new byte[8];
        for (int n = 0; n < 1000000000; n++) {
            for (int i = 0; i < 255; i++) {
                toBitArraySIMD_ge(bs, (byte) i);
            }
        }
    }

    private static void toBitArray_ge(byte[] bs, byte s) {
        bs[7] = (byte) (s >>> 0 & 1);
        bs[6] = (byte) (s >>> 1 & 1);
        bs[5] = (byte) (s >>> 2 & 1);
        bs[4] = (byte) (s >>> 3 & 1);
        bs[3] = (byte) (s >>> 4 & 1);
        bs[2] = (byte) (s >>> 5 & 1);
        bs[1] = (byte) (s >>> 6 & 1);
        bs[0] = (byte) (s >>> 7 & 1);
    }

    private static void toBitArraySIMD_ge(byte[] bs, byte x) {
        val m = 0x2040810204081L;
        long y = ((x & 0xfe) * m | x & 0xff) & M;
        LONG_ARRAY_B.set(bs, 0, y);
    }
}
