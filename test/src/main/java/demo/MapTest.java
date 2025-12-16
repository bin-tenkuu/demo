package demo;

import java.util.HashMap;

/**
 * @author bin
 * @since 2025/12/16
 */
public class MapTest {
    static void main() throws InterruptedException {
        var map = new HashMap<Integer, Integer>();
        for (var i = 0; i < 10; i++) {
            map.put(i, i);
        }
        var threads = new Thread[10];
        for (var i = 0; i < threads.length; i++) {
            var id = i;
            threads[i] = new Thread(() -> {
                for (var n = 0; n < 1000; n++) {
                    for (var j = 0; j < 10; j++) {
                        map.put(j, id);
                    }
                }
            });
            threads[i].start();
        }
        for (var thread : threads) {
            thread.join();
        }
        for (var entry : map.entrySet()) {
            System.out.println(entry.getKey() + " -> " + entry.getValue());
        }
    }
}
