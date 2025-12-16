package demo;

import java.io.IOException;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/07/21
 */
public class SystemInTest {
    public static int globalInt = 0;
    private static boolean readyExit = false;

    public static void main() throws IOException {
        System.out.println("请输入任意字符，回车确定：");
        var in = System.in;
        Thread.ofVirtual().start(new PrintGlobalInt());
        while (true) {
            var newInt = in.read();
            if (newInt == 10) {
                if (readyExit) {
                    System.out.println("退出");
                    return;
                } else {
                    readyExit = true;
                }
            } else {
                readyExit = false;
                globalInt = newInt;
            }

        }
    }

    private static final class PrintGlobalInt implements Runnable {
        private int lastInt;

        @Override
        public void run() {
            while (true) {
                if (lastInt != globalInt) {
                    System.out.println("globalInt = " + globalInt);
                    lastInt = globalInt;
                }
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    System.out.println("printGlobalInt interrupted");
                    return;
                }
            }
        }
    }
}
