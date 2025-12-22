package demo;

import java.util.Scanner;

/**
 * @author bin
 * @since 2025/12/22
 */
public class ThreadTest {
    private static int cache = 1;

    static void main() {
        var thread = new Thread(() -> {
            var in = new Scanner(System.in);
            var i = cache;
            while (!in.next().equals("q")) {
                ++i;
                ++cache;
                var comp = Integer.compare(i, cache);
                switch (comp) {
                    case 0 -> System.out.printf("i: %d, cache: %d\n", i, cache);
                    case -1 -> {
                        System.err.printf("i < cache, i: %d, cache: %d\n", i, cache);
                        i = cache;
                    }
                    case 1 -> {
                        System.err.printf("cache < i, i: %d, cache: %d\n", i, cache);
                        cache = i;
                    }
                }
            }
        });
        thread.start();
    }
}
