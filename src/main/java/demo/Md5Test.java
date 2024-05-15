package demo;

import demo.md5.Md5Calc;
import lombok.val;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.Arrays;

/**
 * @author bin
 * @since 2024/05/15
 */
@SuppressWarnings("preview")
public class Md5Test {

    public static void main() {
        val core = 10;
        val calcs = new Md5Calc[core];
        for (int i = 0; i < core; i++) {
            calcs[i] = new Md5Calc(i, core);
            calcs[i].start();
        }
        show(calcs);
    }

    @SuppressWarnings("BusyWait")
    private static void show(Md5Calc[] md5Calcs) {
        try {
            long time = System.currentTimeMillis();
            while (!Md5Calc.flag) {
                long a = md5Calcs[0].a;
                long b = md5Calcs[0].b;
                for (int i = 1, length = md5Calcs.length; i < length; i++) {
                    Md5Calc md5Calc = md5Calcs[i];
                    if (b > md5Calc.b) {
                        b = md5Calc.b;
                        if (a > md5Calc.a) {
                            a = md5Calc.a;
                        }
                    }
                }
                System.out.println(Arrays.toString(new long[]{a, b}));
                while (!Md5Calc.msgs.isEmpty()) {
                    writeFile(Md5Calc.msgs.removeFirst());
                }
                Thread.sleep(10000);
            }
            System.out.println(STR."time: \{Duration.ofMillis(System.currentTimeMillis() - time)}");
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private static void writeFile(String msg) {
        try (val writer = Files.newBufferedWriter(
                Path.of("./md5.txt"),
                StandardOpenOption.APPEND,
                StandardOpenOption.CREATE
        )) {
            writer.write(msg);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
