package demo;

import demo.test.LongArrayHashMap;
import org.apache.lucene.util.RamUsageEstimator;

import java.util.HashMap;

/**
 * @author bin
 * @since 2024/05/07
 */
@SuppressWarnings("preview")
public class LongArrayHashMapTest {
    public static void main() {
        HashMap<long[], Integer> map = new HashMap<>(100000);
        for (int i = 0; i < 1000000; i++) {
            map.put(new long[]{0, i}, i);
        }
        // System.out.println(ClassLayout.parseInstance(map).toPrintable());
        long size = RamUsageEstimator.sizeOfMap(map);
        System.out.println(STR."map value is \{size}");
        System.out.println(STR."map value is \{RamUsageEstimator.humanReadableUnits(size)}");
        size = RamUsageEstimator.shallowSizeOfInstance(map.getClass());
        System.out.println(STR."arrayMap value is \{size}");
        System.out.println(STR."arrayMap value is \{RamUsageEstimator.humanReadableUnits(size)}");
        System.out.println(map.get(new long[]{0, 255}));


        System.out.println("\n\n\n\n");
        LongArrayHashMap<Integer> arrayMap = new LongArrayHashMap<>(map);
        // System.out.println(ClassLayout.parseInstance(arrayMap).toPrintable());
        size = sizeOfMap(arrayMap);
        System.out.println(STR."arrayMap value is \{size}");
        System.out.println(STR."arrayMap value is \{RamUsageEstimator.humanReadableUnits(size)}");
        size = RamUsageEstimator.shallowSizeOfInstance(arrayMap.getClass());
        System.out.println(STR."arrayMap value is \{size}");
        System.out.println(STR."arrayMap value is \{RamUsageEstimator.humanReadableUnits(size)}");
        System.out.println(arrayMap.get(new long[]{0, 255}));
    }

    private static long sizeOfMap(LongArrayHashMap<?> map) {
        if (map == null) {
            return 0L;
        } else {
            long size = RamUsageEstimator.shallowSizeOf(map);
            long sizeOfEntry = -1L;

            for (var value : map.entrySet()) {
                if (sizeOfEntry == -1L) {
                    sizeOfEntry = RamUsageEstimator.shallowSizeOf(value);
                }

                size += sizeOfEntry;
                size += RamUsageEstimator.sizeOfObject(value.getKey(), 0);
                size += RamUsageEstimator.sizeOfObject(value.getValue(), 0);
            }

            return RamUsageEstimator.alignObjectSize(size);
        }
    }
}
