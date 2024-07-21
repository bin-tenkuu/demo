package demo.ffm;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

@SuppressWarnings("OptionalGetWithoutIsPresent")
public final class Kernel32 {
    private static final Linker linker = Linker.nativeLinker();
    public static final SymbolLookup kernel32 = SymbolLookup.libraryLookup("kernel32.dll", Arena.global());

    /**
     * _Post_equals_last_error_ DWORD GetLastError();
     */
    private static final MethodHandle GetLastError = linker.downcallHandle(
            kernel32.find("GetLastError").get(),
            FunctionDescriptor.of(JAVA_INT));

    public static int GetLastError() throws Throwable {
        return (int) GetLastError.invoke();
    }

    /**
     * long GetCurrentProcessId();
     */
    private static final MethodHandle GetCurrentProcessId = linker.downcallHandle(
            kernel32.find("GetCurrentProcessId").get(),
            FunctionDescriptor.of(JAVA_LONG));

    public static long GetCurrentProcessId() throws Throwable {
        return (long) GetCurrentProcessId.invoke();
    }

    /**
     * HANDLE OpenProcess(
     * long dwDesiredAccess,
     * boolean  bInheritHandle,
     * long dwProcessId
     * );
     */
    private static final MethodHandle OpenProcess = linker.downcallHandle(
            kernel32.find("OpenProcess").get(),
            FunctionDescriptor.of(ADDRESS, JAVA_LONG, JAVA_BOOLEAN, JAVA_LONG));

    public static MemorySegment OpenProcess(
            long dwDesiredAccess, boolean bInheritHandle, long dwProcessId
    ) throws Throwable {
        return (MemorySegment) OpenProcess.invoke(dwDesiredAccess, bInheritHandle, dwProcessId);
    }

    /**
     * void ExitProcess(
     * [in] UINT uExitCode
     * );
     */
    private static final MethodHandle ExitProcess = linker.downcallHandle(
            kernel32.find("ExitProcess").get(),
            FunctionDescriptor.ofVoid(JAVA_INT));

    public static void ExitProcess(int uExitCode) throws Throwable {
        ExitProcess.invoke(uExitCode);
    }

    /**
     * void ExitThread(
     * [in] DWORD dwExitCode
     * );
     */
    private static final MethodHandle ExitThread = linker.downcallHandle(
            kernel32.find("ExitThread").get(),
            FunctionDescriptor.ofVoid(JAVA_INT));

    public static void ExitThread(int dwExitCode) throws Throwable {
        ExitThread.invoke(dwExitCode);
    }

    /**
     * BOOL TerminateThread(
     * [in, out] HANDLE hThread,
     * [in]      DWORD  dwExitCode
     * );
     */
    private static final MethodHandle TerminateThread = linker.downcallHandle(
            kernel32.find("TerminateThread").get(),
            FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS, JAVA_INT));

    public static boolean TerminateThread(MemorySegment hThread, int dwExitCode) throws Throwable {
        return (boolean) TerminateThread.invoke(hThread, dwExitCode);
    }

}
