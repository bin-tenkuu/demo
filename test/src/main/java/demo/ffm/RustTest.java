package demo.ffm;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * @author bin
 * @since 2026/03/11
 */
public class RustTest {
    private static final Linker linker = Linker.nativeLinker();
    public static final SymbolLookup rust = SymbolLookup.libraryLookup("rust/target/debug/librust.so",
            Arena.global());
    /**
     * void hello_world();
     */
    private static final MethodHandle hello_world = linker.downcallHandle(
            rust.find("hello_world").get(),
            FunctionDescriptor.ofVoid());

    /**
     * int addtwo1(int, int)
     */
    private static final MethodHandle addtwo1 = linker.downcallHandle(
            rust.find("addtwo1").get(),
            FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_INT));

    static void main() throws Throwable {
        hello_world.invoke();
        var c = (int) addtwo1.invoke(1, 2);
        System.out.println(c);
    }
}
