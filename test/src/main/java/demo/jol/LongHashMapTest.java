package demo.jol;

import demo.map.LongHashMap;
import org.apache.lucene.util.RamUsageEstimator;

import java.util.HashMap;


public class LongHashMapTest {
    public static void main() {
        var length = 1;
        var map = new HashMap<Long, Long>(length);
        var longMap = new LongHashMap<Long>(length);
        for (long i = 0; i < length; i++) {
            map.put(i, i);
            longMap.put(i, i);
        }
        long size = RamUsageEstimator.sizeOfMap(map);
        System.out.println("map value is " + size);
        System.out.println("map value is " + RamUsageEstimator.humanReadableUnits(size));
        size = sizeOfMap(longMap);
        System.out.println("arrayMap value is " + size);
        System.out.println("arrayMap value is " + RamUsageEstimator.humanReadableUnits(size));
    }

    private static long sizeOfMap(LongHashMap<?> map) {
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
                size += RamUsageEstimator.sizeOfObject(value.getValue(), 0);
            }

            return RamUsageEstimator.alignObjectSize(size);
        }
    }
}
