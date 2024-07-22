package demo.ffm;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

@SuppressWarnings("OptionalGetWithoutIsPresent")
public final class Kernel32 {
    private static final Linker linker = Linker.nativeLinker();
    public static final SymbolLookup kernel32 = SymbolLookup.libraryLookup("kernel32.dll", Arena.global());
    /**
     * typedef [handle] struct
     * {
     * char machine[8];
     * char nmpipe[256];
     * } h_service;
     */
    public static final MemoryLayout Handle = MemoryLayout.structLayout(
            MemoryLayout.sequenceLayout(8, JAVA_CHAR).withName("machine"),
            MemoryLayout.sequenceLayout(256, JAVA_CHAR).withName("nmpipe")
    ).withName("h_service");

    /**
     * _Post_equals_last_error_ DWORD GetLastError();
     */
    private static final MethodHandle GetLastError = linker.downcallHandle(
            kernel32.find("GetLastError").get(),
            FunctionDescriptor.of(JAVA_INT));

    /**
     * @return 返回值是调用线程的最后错误代码。
     * @see <a href="https://learn.microsoft.com/zh-cn/windows/win32/api/errhandlingapi/nf-errhandlingapi-getlasterror">GetLastError 函数 (errhandlingapi.h)</a>
     */
    public static int GetLastError() throws Throwable {
        return (int) GetLastError.invoke();
    }

    /**
     * long GetCurrentProcessId();
     */
    private static final MethodHandle GetCurrentProcessId = linker.downcallHandle(
            kernel32.find("GetCurrentProcessId").get(),
            FunctionDescriptor.of(JAVA_LONG));

    /**
     * @return 返回值是调用进程的进程标识符。
     * @see <a href="https://learn.microsoft.com/zh-cn/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentprocessid">GetCurrentProcessId 函数 (processthreadsapi.h)</a>
     */
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
            FunctionDescriptor.of(ADDRESS,
                    JAVA_LONG, JAVA_BOOLEAN, JAVA_LONG));

    /**
     * @param dwDesiredAccess 对进程对象的访问。 针对进程的安全描述符检查此访问权限。 此参数可以是一个或多个 进程访问权限。
     * @param bInheritHandle 如果此值为 TRUE，则此进程创建的进程将继承句柄。 否则，进程不会继承此句柄。
     * @param dwProcessId 要打开的本地进程的标识符。
     * @return 如果函数成功，则返回值是指定进程的打开句柄。
     * @see <a href="https://learn.microsoft.com/zh-cn/windows/win32/api/processthreadsapi/nf-processthreadsapi-openprocess">OpenProcess 函数 (processthreadsapi.h)</a>
     */
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

    /**
     * @param uExitCode 进程和所有线程的退出代码。
     * @see <a href="https://learn.microsoft.com/zh-cn/windows/win32/api/processthreadsapi/nf-processthreadsapi-exitprocess">ExitProcess 函数 (processthreadsapi.h)</a>
     */
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

    /**
     * @param dwExitCode 线程的退出代码。
     * @see <a href="https://learn.microsoft.com/zh-cn/windows/win32/api/processthreadsapi/nf-processthreadsapi-exitthread">ExitThread 函数 (processthreadsapi.h)</a>
     */
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

    /**
     * @param hThread 要终止的线程的句柄。
     * @param dwExitCode 线程的退出代码。
     * @return 如果该函数成功，则返回值为非零值。
     * @see <a href="https://learn.microsoft.com/zh-cn/windows/win32/api/processthreadsapi/nf-processthreadsapi-terminatethread">TerminateThread 函数 (processthreadsapi.h)</a>
     */
    public static boolean TerminateThread(MemorySegment hThread, int dwExitCode) throws Throwable {
        return (boolean) TerminateThread.invoke(hThread, dwExitCode);
    }

    /**
     * DWORD GetPriorityClass(
     * [in] HANDLE hProcess
     * );
     */
    private static final MethodHandle GetPriorityClass = linker.downcallHandle(
            kernel32.find("GetPriorityClass").get(),
            FunctionDescriptor.of(JAVA_INT, ADDRESS.withTargetLayout(Handle)));

    /**
     * @param hProcess 进程的句柄。
     * @return 如果函数成功，则返回值是指定进程的优先级类。
     * @see <a href="https://learn.microsoft.com/zh-cn/windows/win32/api/processthreadsapi/nf-processthreadsapi-getpriorityclass">GetPriorityClass 函数 (processthreadsapi.h)</a>
     */
    public static int GetPriorityClass(MemorySegment hProcess) throws Throwable {
        return (int) GetPriorityClass.invoke(hProcess);
    }

    /**
     * void GetCurrentThreadStackLimits(
     * [out] PULONG_PTR LowLimit,
     * [out] PULONG_PTR HighLimit
     * );
     */
    private static final MethodHandle GetCurrentThreadStackLimits = linker.downcallHandle(
            kernel32.find("GetCurrentThreadStackLimits").get(),
            FunctionDescriptor.ofVoid(ADDRESS.withTargetLayout(JAVA_LONG), ADDRESS.withTargetLayout(JAVA_LONG)));

    /**
     * @param LowLimit 一个指针变量，用于接收当前线程堆栈的下边界。
     * @param HighLimit 一个指针变量，用于接收当前线程堆栈的上边界。
     * @see <a href="https://learn.microsoft.com/zh-cn/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentthreadstacklimits">GetCurrentThreadStackLimits 函数 (processthreadsapi.h)</a>
     */
    public static void GetCurrentThreadStackLimits(MemorySegment LowLimit, MemorySegment HighLimit) throws Throwable {
        GetCurrentThreadStackLimits.invoke(LowLimit, HighLimit);
    }

    /**
     * BOOL CloseHandle(
     * [in] HANDLE hObject
     * );
     */
    private static final MethodHandle CloseHandle = linker.downcallHandle(
            kernel32.find("CloseHandle").get(),
            FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS));

    /**
     * @param hObject 打开对象的有效句柄。
     * @return 如果该函数成功，则返回值为非零值。
     * @see <a href="https://learn.microsoft.com/zh-cn/windows/win32/api/handleapi/nf-handleapi-closehandle">CloseHandle 函数 (handleapi.h)</a>
     */
    public static boolean CloseHandle(MemorySegment hObject) throws Throwable {
        return (boolean) CloseHandle.invoke(hObject);
    }

    /**
     * BOOL ReadProcessMemory(
     * [in]  HANDLE  hProcess,
     * [in]  LPCVOID lpBaseAddress,
     * [out] LPVOID  lpBuffer,
     * [in]  SIZE_T  nSize,
     * [out] SIZE_T  *lpNumberOfBytesRead
     * );
     */
    private static final MethodHandle ReadProcessMemory = linker.downcallHandle(
            kernel32.find("ReadProcessMemory").get(),
            FunctionDescriptor.of(JAVA_BOOLEAN,
                    ADDRESS.withTargetLayout(Handle),
                    JAVA_LONG,
                    ADDRESS,
                    JAVA_LONG,
                    ADDRESS.withTargetLayout(JAVA_LONG)));

    /**
     * @param hProcess 包含正在读取的内存的进程句柄。 句柄必须具有对进程的PROCESS_VM_READ访问权限。
     * @param lpBaseAddress 指向从中读取的指定进程中基址的指针。 在进行任何数据传输之前，系统会验证指定大小的基址和内存中的所有数据是否可供读取访问，如果无法访问，则函数将失败。
     * @param lpBuffer 指向从指定进程的地址空间接收内容的缓冲区的指针。
     * @param nSize 要从指定进程读取的字节数。
     * @param lpNumberOfBytesRead 指向变量的指针，该变量接收传输到指定缓冲区的字节数。 如果 lpNumberOfBytesRead 为 NULL，则忽略 参数。
     * @return 如果该函数成功，则返回值为非零值。
     * @see <a href="https://learn.microsoft.com/zh-cn/windows/win32/api/memoryapi/nf-memoryapi-readprocessmemory">ReadProcessMemory 函数 (memoryapi.h)</a>
     */
    public static boolean ReadProcessMemory(
            MemorySegment hProcess,
            long lpBaseAddress,
            MemorySegment lpBuffer,
            long nSize,
            MemorySegment lpNumberOfBytesRead
    ) throws Throwable {
        return (boolean) ReadProcessMemory.invoke(hProcess, lpBaseAddress, lpBuffer, nSize, lpNumberOfBytesRead);
    }
}
