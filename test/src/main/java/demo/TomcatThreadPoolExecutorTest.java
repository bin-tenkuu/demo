package demo;

import demo.util.Threads;

public class TomcatThreadPoolExecutorTest {
    static void main() throws InterruptedException {
        for (int i = 0; i < 100; i++) {
            addRun();
            Threads.debug();
        }
        for (int i = 0; i < 10; i++) {
            Thread.sleep(3000);
            addRun();
            Threads.debug();
        }
        Thread.sleep(1000);
        Threads.debug();
    }

    private static void addRun() {
        Threads.execute(() -> {
            Threads.debug();
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

}
