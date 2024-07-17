package demo;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.val;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/07/17
 */
public class MethodHandlerTest {
    @NoArgsConstructor
    @AllArgsConstructor
    @ToString
    private static final class NormalClass {
        public int a;
        public String b;
    }

    public record RecordClass(int a, String b) {
    }

    private static final MethodHandles.Lookup lookup = MethodHandles.lookup();

    public static void main() throws Throwable {
        testNoArg(NormalClass.class);
        test2Arg(NormalClass.class, 1, "2");
        test2Arg(RecordClass.class, 1, "2");
    }

    private static void testNoArg(Class<?> clazz) throws Throwable {
        val methodType = MethodType.methodType(void.class);
        val constructor = lookup.findConstructor(clazz, methodType);
        val object = constructor.invoke();
        System.out.println(object);
        val getA = clazz.getDeclaredField("a");
        getA.setAccessible(true);
        val getter = lookup.unreflectGetter(getA);
        System.out.println(getter.invoke(object));
        val varHandle = lookup.unreflectVarHandle(getA);
        System.out.println(varHandle.get(object));
    }

    private static void test2Arg(Class<?> clazz, int argA, String argB) throws Throwable {
        val methodType = MethodType.methodType(void.class, int.class, String.class);
        val constructor = lookup.findConstructor(clazz, methodType);
        val object = constructor.invoke(argA, argB);
        System.out.println(object);
        val getA = clazz.getDeclaredField("a");
        getA.setAccessible(true);
        val getter = lookup.unreflectGetter(getA);
        val a = getter.invoke(object);
        System.out.println(a);
        val varHandle = lookup.unreflectVarHandle(getA);
        System.out.println(varHandle.get(object));
    }
}
