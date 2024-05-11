package demo;

import jdk.incubator.vector.*;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.results.format.ResultFormatType;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@State(Scope.Benchmark)
@Fork(1)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 3)
public class CountJmhTest {
    // 1g
    private static final int arraySize = 1 << 30;

    // @Param({"1"})
    public int dataLength = 2;
    private byte[][] datas;
    private long sum;

    @Setup(Level.Iteration)
    public void prepare() {
        final Random rnd = new Random(0);
        datas = new byte[dataLength][arraySize];
        for (int i = 0; i < dataLength; i++) {
            rnd.nextBytes(datas[i]);
        }
    }

    @TearDown(Level.Iteration)
    public void check() {
        System.out.println(sum);
    }

    public static void main() {
        Options opt = new OptionsBuilder()
                .include(CountJmhTest.class.getSimpleName())
                .result("result.json")
                .resultFormat(ResultFormatType.JSON)
                .build();
        try {
            new Runner(opt).run();
        } catch (RunnerException e) {
            throw new RuntimeException(e);
        }

        // java.util.function.BiConsumer<String, java.util.function.Consumer<Main>> consumer = (name, func) -> {
        //     Main main = new Main();
        //     main.prepare();
        //     long start = System.nanoTime();
        //     func.accept(main);
        //     System.out.println((System.nanoTime() - start) / 1000000000.0);
        //     System.out.println(STR."\{name}\tsum = \{main.sum}");
        // };
        // consumer.accept("sample", Main::sample);
        // consumer.accept("noIf", Main::noIf);
        // consumer.accept("withSort", Main::withSort);
        // consumer.accept("withSortAndBranchPrediction", Main::withSortAndBranchPrediction);
        // consumer.accept("withSimdNoIf", Main::withSimdNoIf);
        // consumer.accept("withSimdAndIf", Main::withSimdAndIf);
    }

    // @Benchmark
    public void sample() {
        long sum = 0;
        for (int i = 0; i < dataLength; i++) {
            byte[] bytes = datas[i];
            for (int j = 0; j < arraySize; ++j) {
                if (bytes[j] >= 0) {
                    sum += bytes[j];
                }
            }
        }
        this.sum = sum;
    }

    // @Benchmark
    public void withSort() {
        long sum = 0;
        for (int i = 0; i < dataLength; i++) {
            byte[] data = datas[i];
            Arrays.sort(data);
            for (int j = 0; j < arraySize; ++j) {
                if (data[j] >= 0) {
                    sum += data[j];
                }
            }
        }
        this.sum = sum;
    }

    // @Benchmark
    public void withSortAndSearch() {
        long sum = 0;
        for (int i = 0; i < dataLength; i++) {
            byte[] data = datas[i];
            Arrays.sort(data);
            for (int j = Arrays.binarySearch(data, (byte) 0); j < arraySize; ++j) {
                sum += data[j];
            }
        }
        this.sum = sum;
    }

    // @Benchmark
    public void withSortAndBranchPrediction() {
        long sum = 0;
        for (int i = 0; i < dataLength; i++) {
            byte[] data = datas[i];
            Arrays.sort(data);
            int j = 0;
            for (; j < arraySize; ++j) {
                if (data[j] > 0) {
                    break;
                }
            }
            for (; j < arraySize; ++j) {
                sum += data[j];
            }
        }
        this.sum = sum;
    }

    // @Benchmark
    public void noIf() {
        long sum = 0;
        for (int i = 0; i < dataLength; i++) {
            byte[] data = datas[i];
            for (int j = 0; j < arraySize; ++j) {
                int t = data[j] >> 31;
                sum += ~t & data[j];
            }
        }
        this.sum = sum;
    }

    @Benchmark
    public void withSimdNoIf() {
        VectorSpecies<Byte> SPECIES = ByteVector.SPECIES_MAX;
        final int length = SPECIES.length();

        long sum = 0;
        final byte[] bytes = new byte[length];
        for (int i = 0; i < dataLength; i++) {
            byte[] data = datas[i];

            for (int j = 0; j < data.length; j += length) {
                var m = SPECIES.indexInRange(i, data.length);
                ByteVector tmp = ByteVector.fromArray(SPECIES, data, j, m)
                        .lanewise(VectorOperators.ASHR, 7)
                        .not()
                        .and(ByteVector.fromArray(SPECIES, data, j, m));
                tmp.intoArray(bytes, 0);
                sum += sum(bytes);
            }
        }

        this.sum = sum;
    }

    @Benchmark
    public void withSimdAndIf() {
        VectorSpecies<Byte> SPECIES = ByteVector.SPECIES_MAX;
        final int length = SPECIES.length();

        long sum = 0;
        final byte[] bytes = new byte[length];
        for (int i = 0; i < dataLength; i++) {
            byte[] data = datas[i];

            for (int j = 0; j < data.length; j += length) {
                var m = SPECIES.indexInRange(i, data.length);
                ByteVector tmp = ByteVector.fromArray(SPECIES, data, j, m);
                VectorMask<Byte> compare = tmp.compare(VectorOperators.LT, 0);
                tmp = tmp.blend(SPECIES.zero(), compare);
                tmp.intoArray(bytes, 0);
                sum += sum(bytes);
            }
        }

        this.sum = sum;
    }

    private static long sum(byte[] bs) {
        long sum = 0;
        for (byte b : bs) {
            sum += b;
        }
        return sum;
    }

    // @Benchmark
    public void noOp() {
        this.sum = 68183805086L;
    }

}
