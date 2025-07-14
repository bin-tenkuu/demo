package demo.util;

import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.concurrent.*;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/12/12
 */
@SuppressWarnings("unused")
@Slf4j
public class Threads {
    private static final int core = Runtime.getRuntime().availableProcessors();
    private static final ThreadGroup group = new ThreadGroup("Threads");
    private static int threadInitNumber = 0;
    private static final ThreadFactory threadFactory = r -> {
        val thread = new Thread(group, r, "Threads-" + threadInitNumber++);
        thread.setDaemon(true);
        return thread;
    };
    private static final TaskQueue taskqueue = new TaskQueue();
    private static final RejectedExecutionHandler handler = (r, executor) -> {
        try {
            executor.getQueue().put(r);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    };
    public static final ThreadPoolExecutor executor = new ThreadPoolExecutor(
            1, core * 2 - 1,
            1, TimeUnit.MINUTES,
            taskqueue,
            threadFactory,
            handler
    );
    public static final ScheduledExecutorService SCHEDULE = new ScheduledThreadPoolExecutor(1, threadFactory, handler);
    private static final ConcurrentHashMap<Object, Future<?>> DELAYED_MAP = new ConcurrentHashMap<>();

    public static void debug() {
        System.out.printf("核心/最大线程数:%s/%s	活动线程数:%d	当前排队线程数:%d%n",
                executor.getCorePoolSize(), executor.getMaximumPoolSize(), executor.getActiveCount(), taskqueue.size()
        );
    }

    public static Future<?> submit(Runnable task) {
        // log.info("当前排队任务数submit：{}", taskqueue.size());
        return executor.submit(task);
    }

    public static void execute(@NotNull Runnable command) {
        // log.info("当前排队任务数execute：{}", taskqueue.size());
        executor.execute(command);
    }

    public static boolean remove(Runnable task) {
        return executor.remove(task);
    }

    public static void waitList(List<Future<?>> list) throws ExecutionException, InterruptedException {
        for (Future<?> future : list) {
            future.get();
        }
    }

    /**
     * 防抖函数，固定延时函数,
     * 用于防止短时间内多次触发，也用于固定延时时间内只执行一次,
     * 如：表变化消息，连接断开消息
     */
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

    /**
     * 优先开新线程，然后放入队列
     */
    @Setter
    public static class TaskQueue extends LinkedBlockingQueue<Runnable> {

        public boolean offer(@NotNull Runnable o) {
            if (size() == 0) {
                return super.offer(o);
            } else {
                return false;
            }
        }
    }

}
