package demo.util;

import java.util.concurrent.*;

/**
 * 防抖函数，固定延时函数
 * <p>
 * 用于防止短时间内多次触发，也用于固定延时时间内只执行一次
 * <p>
 * 如：表变化消息，连接断开消息
 *
 * @author bin
 * @since 2023/03/17
 */
@SuppressWarnings("unused")
public class DebounceUtil {
    private static final ScheduledExecutorService SCHEDULE = Executors.newSingleThreadScheduledExecutor();
    private static final ConcurrentHashMap<Object, Future<?>> DELAYED_MAP = new ConcurrentHashMap<>();

    public static void debounce(final Object key, final Runnable runnable, long delay, TimeUnit unit) {
        final Future<?> prev = DELAYED_MAP.put(key, SCHEDULE.schedule(new Command(key, runnable), delay, unit));
        if (prev != null) {
            prev.cancel(true);
        }
    }

    public static void shutdown() {
        SCHEDULE.shutdownNow();
    }

    private static class Command implements Runnable {
        private final Object key;
        private final Runnable runnable;

        private Command(Object key, Runnable runnable) {
            this.key = key;
            this.runnable = runnable;
        }

        @Override
        public void run() {
            try {
                runnable.run();
            } finally {
                DELAYED_MAP.remove(key);
            }
        }
    }

    public static void main(String[] args) {
        DebounceUtil.debounce("1", () -> System.out.println(11), 3, TimeUnit.SECONDS);
        DebounceUtil.debounce("1", () -> System.out.println(22), 3, TimeUnit.SECONDS);
        DebounceUtil.debounce("1", () -> System.out.println(33), 3, TimeUnit.SECONDS);
        DebounceUtil.debounce("2", () -> System.out.println(44), 3, TimeUnit.SECONDS);
        DebounceUtil.debounce("2", () -> System.out.println(44), 3, TimeUnit.SECONDS);
        DebounceUtil.debounce("2", () -> System.out.println(44), 3, TimeUnit.SECONDS);
    }
}
