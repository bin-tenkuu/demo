package demo;

import demo.md5.Md5Calc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * @author bin
 * @since 2024/05/15
 */
@SuppressWarnings({"BusyWait"})
public class Md5Test {
    private static final Md5Calc[] calcs = new Md5Calc[10];

    public static void main() {
        int core = calcs.length;
        for (var i = 0; i < core; i++) {
            calcs[i] = new Md5Calc(i, core);
            calcs[i].start();
        }
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            Md5Calc.flag = true;
            show();
        }));
        try {
            while (!Md5Calc.flag) {
                show();
                Thread.sleep(10000);
            }
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private static void show() {
        var min = calcs[0];
        var max = calcs[0];
        for (int i = 1, length = calcs.length; i < length; i++) {
            var calc = calcs[i];
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
    }

    private static void writeFile(String msg) {
        try (var writer = Files.newBufferedWriter(
                Path.of("./md5.txt"),
                StandardOpenOption.APPEND
        )) {
            writer.write(msg);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
