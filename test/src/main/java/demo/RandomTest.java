package demo;

import org.apache.commons.math3.random.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.IntSupplier;
import java.util.function.Supplier;

public class RandomTest {
    private static final int MAX_X = 1 << 12;
    private static final int MAX_Y = 1 << 12;
    private static final ExecutorService executor = Executors.newFixedThreadPool(
            Runtime.getRuntime().availableProcessors()
    );
    private static final int SEED = 0;
    private static final int[] SEEDS = {0, 0};

    public static void main() {
        final RandomType[] gens = {
                new RandomType("rgb-Random", () -> new Random(SEED)),
                new RandomType("rgb-SecureRandom", () -> new SecureRandom(new byte[0])),
                new RandomType("rgb-ISAACRandom", () -> new RandomAdaptor(new ISAACRandom(SEEDS))),
                new RandomType("rgb-MersenneTwister", () -> new RandomAdaptor(new MersenneTwister(SEEDS))),
                new RandomType("rgb-Well512a", () -> new RandomAdaptor(new Well512a(SEEDS))),
                new RandomType("rgb-Well1024a", () -> new RandomAdaptor(new Well1024a(SEEDS))),
                new RandomType("rgb-Well19937a", () -> new RandomAdaptor(new Well19937a(SEEDS))),
                new RandomType("rgb-Well19937c", () -> new RandomAdaptor(new Well19937c(SEEDS))),
                new RandomType("rgb-Well44497a", () -> new RandomAdaptor(new Well44497a(SEEDS))),
                new RandomType("rgb-Well44497b", () -> new RandomAdaptor(new Well44497b(SEEDS))),
        };
        final List<Future<?>> list = new ArrayList<>(gens.length * 4);

        for (RandomType type : gens) {
            list.add(executor.submit(() -> toImage(type, 0)));
            list.add(executor.submit(() -> toImage(type, 1)));
            list.add(executor.submit(() -> toImage(type, 2)));
            list.add(executor.submit(() -> toImage(type, 3)));
        }
        while (!list.isEmpty()) {
            if (list.getFirst().isDone()) {
                list.removeFirst();
            }
            Thread.yield();
        }
        executor.shutdown();
    }

    private static void toImage(final RandomType type, final int i) {
        Generator gen = new Generator(type);
        switch (i) {
            case 0 -> toImage(type.name + "-1Bit", gen::get1Bit);
            case 1 -> toImage(type.name + "-8Bit", gen::get8Bit);
            case 2 -> toImage(type.name + "-24Bit", gen::get24Bit);
            case 3 -> toImage(type.name + "-32Bit", gen::get32Bit);
            default -> throw new IllegalArgumentException("" + i);
        }
    }

    private static void toImage(final String name, final IntSupplier getRGB) {
        BufferedImage image = new BufferedImage(MAX_X, MAX_Y, BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < MAX_X; x++) {
            for (int y = 0; y < MAX_Y; y++) {
                image.setRGB(x, y, getRGB.getAsInt());
            }
        }
        try {
            File file = new File(name + ".jpg");
            System.out.println("Writing to " + file.getAbsolutePath());

            ImageIO.write(image, "jpg", file);
            if (!file.isFile()) {
                System.err.println("Failed to write to " + file.getAbsolutePath());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private record RandomType(String name, Supplier<Random> supplier) {
    }

    private static final class Generator {
        private static final int min = 0x000000;
        private static final int max = 0xFFFFFF;
        private final Random random;
        private int count = 32;
        private int n = 0;

        private Generator(RandomType type) {
            this.random = type.supplier.get();
        }

        public int get1Bit() {
            if (count == 32) {
                count = 0;
                n = random.nextInt();
            }
            int i = n & 0x1;
            count += 1;
            n = n >> 1;
            if (i == 1) {
                return min;
            } else {
                return max;
            }
        }

        public int get8Bit() {
            if (count == 32) {
                count = 0;
                n = random.nextInt();
            }
            int i = n & 0xFF;
            count += 8;
            n = n >> 8;
            return i << 16 | i << 8 | i;
        }

        public int get24Bit() {
            if (count == 32) {
                count = 0;
                n = random.nextInt();
            }
            int i;
            // 0, 8, 16, 24
            if (count <= 8) {
                i = n & 0xFFFFFF;
                n = n >> 24;
                count += 24;
            } else {
                int oldBit = 32 - count;
                i = n & (1 << oldBit);
                n = random.nextInt();
                count = 24 - oldBit;
                i = (i << oldBit) | (n & (1 << count));
                n = n >> count;
            }
            return i;
        }

        public int get32Bit() {
            return random.nextInt();
        }

    }
}
