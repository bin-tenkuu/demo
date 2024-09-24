package demo;

import demo.threadpool.TaskQueue;
import lombok.val;

import java.util.concurrent.*;

public class TomcatThreadPoolExecutorTest {
    private static final ThreadGroup group = Thread.currentThread().getThreadGroup();
    private static final ThreadFactory threadFactory = r -> {
        val thread = new Thread(group, r);
        thread.setDaemon(true);
        return thread;
    };
    private static final TaskQueue taskqueue = new TaskQueue(40);
    private static final ThreadPoolExecutor executor = new ThreadPoolExecutor(
            1, 10,
            0, TimeUnit.SECONDS,
            taskqueue, threadFactory
    );

    public static void main() throws InterruptedException {
        taskqueue.setParent(executor);
        for (int i = 0; i < 20; i++) {
            addRun();
        }
        Thread.sleep(3000);
        addRun();
        Thread.sleep(1000);
    }

    private static void addRun() {
        executor.execute(() -> {
            logStatus();
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    private static void logStatus() {
        val size = taskqueue.size();
        val capacity = taskqueue.remainingCapacity();
        System.out.printf("核心线程数:%s	活动线程数:%2d/%s	当前排队线程数:%2d + %2d = %s%n",
                executor.getCorePoolSize(), executor.getActiveCount(), executor.getMaximumPoolSize(),
                size, capacity, size + capacity
        );
    }

}
