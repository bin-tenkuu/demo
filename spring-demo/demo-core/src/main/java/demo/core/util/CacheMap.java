package demo.core.util;

import java.time.Duration;
import java.util.HashMap;

/// @author bin
/// @since 2023/08/22
public class CacheMap<K, V> {
    public static final long DEFAULT_TIMEOUT = Duration.ofMinutes(10).toMillis();
    /// 过期时间,毫秒
    private final long timeout;

    private final HashMap<K, Node> map = new HashMap<>();

    /// @param timeout 过期时间,毫秒
    public CacheMap(long timeout) {
        this.timeout = timeout;
    }

    public CacheMap() {
        this(DEFAULT_TIMEOUT);
    }

    private long nextExpirationTime = Long.MAX_VALUE;

    private void expungeExpiredEntries() {
        var time = System.currentTimeMillis();
        if (nextExpirationTime > time) {
            return;
        }
        nextExpirationTime = Long.MAX_VALUE;
        var iterator = map.values().iterator();
        while (iterator.hasNext()) {
            var v = iterator.next();
            if (v.isBeOverdue(time)) {
                iterator.remove();
            } else if (nextExpirationTime > v.time) {
                nextExpirationTime = v.time;
            }
        }
    }

    public int getSize() {
        expungeExpiredEntries();
        return map.size();
    }

    public void clear() {
        map.clear();
        nextExpirationTime = Long.MAX_VALUE;
    }

    public void set(K key, V value, long timeout) {
        expungeExpiredEntries();
        var node = new Node(value, timeout);
        map.put(key, node);
        if (this.nextExpirationTime > node.time) {
            this.nextExpirationTime = node.time;
        }
    }

    public void set(K key, V value) {
        set(key, value, timeout);
    }

    public V get(K key) {
        var node = map.get(key);
        if (node == null) {
            return null;
        }
        if (node.isBeOverdue()) {
            map.remove(key);
            return null;
        }
        return node.v;
    }

    public boolean contains(K key) {
        var node = map.get(key);
        if (node == null) {
            return false;
        }
        if (node.isBeOverdue()) {
            map.remove(key);
            return false;
        }
        return true;
    }

    public V remove(K key) {
        var node = map.remove(key);
        if (node == null) {
            return null;
        }
        if (node.isBeOverdue()) {
            return null;
        }
        return node.v;
    }

    private class Node {
        private final V v;

        private final long time;

        public Node(V v, long timeout) {
            this.v = v;
            this.time = timeout + System.currentTimeMillis();
        }

        public Node(V v) {
            this(v, 0);
        }

        public boolean isBeOverdue(long time) {
            return time >= this.time;
        }

        public boolean isBeOverdue() {
            return System.currentTimeMillis() >= this.time;
        }

        @Override
        public String toString() {
            return String.format("%s:%s", isBeOverdue() ? "timeout" : "waiting", v);
        }
    }
}
