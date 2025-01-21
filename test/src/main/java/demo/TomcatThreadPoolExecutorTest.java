package demo;

import demo.util.Threads;
import lombok.val;

public class TomcatThreadPoolExecutorTest {
    public static void main() throws InterruptedException {
        for (int i = 0; i < 100; i++) {
            addRun();
        }
        for (int i = 0; i < 10; i++) {
            Thread.sleep(3000);
            addRun();
        }
        Thread.sleep(1000);
    }

    private static void addRun() {
        Threads.execute(() -> {
            logStatus();
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    private static void logStatus() {
        val executor = Threads.executor;
        val taskqueue = executor.getQueue();
        val size = taskqueue.size();
        System.out.printf("核心线程数:%s	活动线程数:%2d/%s	当前排队线程数:%2d%n",
                executor.getCorePoolSize(), executor.getActiveCount(), executor.getMaximumPoolSize(), size
        );
    }

}
