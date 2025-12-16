package demo.ffm;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/07/21
 */
public class WindowsApiTest {
    private static final Arena arena = Arena.ofAuto();
    private static final long PROCESS_ALL_ACCESS = 0x1FFFFF;

    public static void main() throws Throwable {
        test();
        testExitThread();
    }

    private static void test() throws Throwable {
        System.out.println(Kernel32.GetLastError());
        var currentProcessId = Kernel32.GetCurrentProcessId();
        System.out.println(currentProcessId);
        var handle = Kernel32.OpenProcess(PROCESS_ALL_ACCESS, false, currentProcessId);
        System.out.println(Integer.toHexString(Kernel32.GetPriorityClass(handle)));
        System.out.println(Kernel32.GetLastError());
        var low = arena.allocate(ValueLayout.JAVA_LONG);
        var high = arena.allocate(ValueLayout.JAVA_LONG);
        Kernel32.GetCurrentThreadStackLimits(low, high);
        System.out.println("LowLimit:" + low.get(ValueLayout.JAVA_LONG, 0));
        System.out.println("HighLimit" + high.get(ValueLayout.JAVA_LONG, 0));
        System.out.println(Kernel32.CloseHandle(handle));
    }

    private static void testExitThread() throws Throwable {
        Kernel32.ExitThread(0);
        // 将永远不会执行下面的代码
        System.out.println("ExitThread");
    }

}
