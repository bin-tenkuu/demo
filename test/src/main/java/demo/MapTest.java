package demo;

import java.util.*;

/**
 * @author bin
 * @since 2025/12/16
 */
public class MapTest {
    private static Set<String> keys;
    // 实际场景这里是 AtomicReference<HashMap<String, Integer>>
    private static Map<String, Integer> map;

    public static void main() throws InterruptedException {
        var hashArr = new String[]{"Aa", "BB"};
        // size=32-1
        var size = 0b11111;
        var bit = 5;
        var keyList = new HashSet<String>();
        // hash 正常分布 31 个
        for (var i = 0; i < size; i++) {
            var s = String.valueOf(i);
            keyList.add(s);
            System.out.printf("hash %s = %s\n", s, s.hashCode());
        }
        map = buildMap(keyList);
        keys = keyList;
        var threads = new Thread[16];
        for (var i = 0; i < threads.length; i++) {
            var id = i;
            threads[i] = new Thread(() -> {
                while (true) {
                    for (var s : keys) {
                        map.put(s, id);
                    }
                    Thread.yield();
                }
            });
            threads[i].setDaemon(true);
            threads[i].start();
        }
        Thread.sleep(1000);
        // 第一次读取
        var newKeys = new HashSet<String>();
        // 更新keys: hash 碰撞分布 31 个
        for (var i = 0; i < size; i++) {
            var s = generateKey(i, hashArr, bit);
            newKeys.add(s);
            System.out.printf("hash %s = %s\n", s, s.hashCode());
        }
        readMap(newKeys);
        Thread.sleep(1000);
        // 第二次读取
        readMap(null);
        Thread.sleep(1000);
        // 第三次读取
        readMap(null);
    }

    private static void readMap(Collection<String> newKeys) {
        Set<String> keysSnapshot;
        if (newKeys == null) {
            keysSnapshot = keys;
        } else {
            keysSnapshot = new HashSet<>(keys);
            keysSnapshot.addAll(newKeys);
        }
        var newMap = buildMap(keysSnapshot);
        var mapSnapshot = map;
        map = newMap;
        keys = keysSnapshot;
        new Thread(() -> {
            for (var entry : mapSnapshot.entrySet()) {
                System.out.println(entry.getKey() + " -> " + entry.getValue());
            }
        }).start();
    }

    private static HashMap<String, Integer> buildMap(Collection<String> list) {
        var map = new HashMap<String, Integer>();
        for (var s : list) {
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
