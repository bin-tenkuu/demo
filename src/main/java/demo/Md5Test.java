package demo;

import demo.md5.Md5Calc;
import lombok.val;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * @author bin
 * @since 2024/05/15
 */
// @SuppressWarnings("preview")
@SuppressWarnings({"InfiniteLoopStatement", "BusyWait"})
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

    private static void show(Md5Calc[] md5Calcs) {
        try {
            // long time = System.currentTimeMillis();
            while (true) {
                Md5Calc min = md5Calcs[0];
                Md5Calc max = md5Calcs[0];
                for (int i = 1, length = md5Calcs.length; i < length; i++) {
                    Md5Calc calc = md5Calcs[i];
                    if (min.compareTo(calc) > 0) {
                        min = calc;
                    } else if (max.compareTo(calc) < 0) {
                        max = calc;
                    }
                }
                System.out.print(min);
                System.out.print(" ~ ");
                System.out.println(max);
                while (!Md5Calc.msgs.isEmpty()) {
                    writeFile(Md5Calc.msgs.removeFirst());
                }
                Thread.sleep(10000);
            }
            // System.out.println(STR."time: \{Duration.ofMillis(System.currentTimeMillis() - time)}");
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
