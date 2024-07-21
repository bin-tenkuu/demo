package demo;

import demo.ffm.Kernel32;
import lombok.val;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/07/21
 */
public class WindowsApiTest {
    // private static final Arena arena = Arena.ofAuto();
    private static final long PROCESS_ALL_ACCESS = 0x1FFFFF;

    public static void main() throws Throwable {
        test();
        testExitThread();
    }

    private static void test() throws Throwable {
        System.out.println(Kernel32.GetLastError());
        val currentProcessId = Kernel32.GetCurrentProcessId();
        System.out.println(currentProcessId);
        val memorySegment = Kernel32.OpenProcess(PROCESS_ALL_ACCESS, false, currentProcessId);
        System.out.println(memorySegment);
        System.out.println(Kernel32.GetLastError());
    }

    private static void testExitThread() throws Throwable {
        Kernel32.ExitThread(0);
        // 将永远不会执行下面的代码
        System.out.println("ExitThread");
    }

}
