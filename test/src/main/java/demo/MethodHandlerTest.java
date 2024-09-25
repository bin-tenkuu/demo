package demo;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.val;

import java.lang.invoke.MethodHandles;

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
        test(NormalClass.class);
        test(NormalClass.class);
        test(RecordClass.class);
    }

    private static void test(Class<?> clazz) throws Throwable {
        val clazzConstructor = clazz.getConstructors()[0];
        clazzConstructor.setAccessible(true);
        val count = clazzConstructor.getParameterCount();
        val constructor = lookup.unreflectConstructor(clazzConstructor);
        final Object object;
        if (count == 0) {
            object = constructor.invoke();
        } else if (count == 2) {
            object = constructor.invoke(1, "2");
        } else {
            throw new IllegalArgumentException("count is not 0 or 2");
        }
        System.out.println(object);
        val getA = clazz.getDeclaredField("a");
        getA.setAccessible(true);
        val getter = lookup.unreflectGetter(getA);
        System.out.println(getter.invoke(object));
        val varHandle = lookup.unreflectVarHandle(getA);
        System.out.println(varHandle.get(object));
    }
}
