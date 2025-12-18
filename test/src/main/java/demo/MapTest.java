package demo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * @author bin
 * @since 2025/12/16
 */
public class MapTest {
    private static final List<String> keys = new ArrayList<>();
    private static HashMap<String, Integer> map;

    public static void main() throws InterruptedException {
        var hashArr = new String[]{"Aa", "BB"};
        // size=32-1
        var size = 0b11111;
        var bit = 5;
        // hash 正常分布 32 个
        for (var i = 0; i <= size; i++) {
            var s = String.valueOf(i);
            keys.add(s);
            // System.out.printf("hash %s = %s\n", s, s.hashCode());
        }
        map = buildMap();
        var threads = new Thread[16];
        for (var i = 0; i < threads.length; i++) {
            var id = i;
            threads[i] = new Thread(() -> {
                while (true) {
                    // noinspection ForLoopReplaceableByForEach
                    for (int n = 0; n < keys.size(); n++) {
                        var s = keys.get(n);
                        map.computeIfPresent(s, (_, _) -> id);
                    }
                    Thread.yield();
                }
            });
            threads[i].setDaemon(true);
            threads[i].start();
        }
        Thread.sleep(1000);
        // 第一次读取
        // 更新keys: hash 碰撞分布 32 个
        for (var i = 0; i <= size; i++) {
            var s = generateKey(i, hashArr, bit);
            keys.add(s);
            // System.out.printf("hash %s = %s\n", s, s.hashCode());
        }
        readMap();
        Thread.sleep(1000);
        // 第二次读取
        // 删除 key: hash 正常分布 32 个
        keys.subList(0, size + 1).clear();
        readMap();
        Thread.sleep(1000);
        // 第三次读取
        readMap();
    }

    private static void readMap() {
        var newMap = buildMap();
        var mapSnapshot = map;
        map = newMap;
        new Thread(() -> {
            for (var entry : mapSnapshot.entrySet()) {
                System.out.println(entry.getKey() + " -> " + entry.getValue());
            }
        }).start();
    }

    private static HashMap<String, Integer> buildMap() {
        var map = new HashMap<String, Integer>(keys.size());
        // 这边是同步的，所以可以用 for-each
        for (var s : keys) {
            map.put(s, -1);
        }
        return map;
    }

    private static String generateKey(int i, String[] hashArr, int bit) {
        var sb = new StringBuilder();
        int n = i;
        for (int j = 0; j < bit; j++) {
            sb.append(hashArr[n & 1]);
            n >>= 1;
        }
        return sb.toString();
    }
}
