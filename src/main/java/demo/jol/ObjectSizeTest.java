package demo.jol;

import org.apache.lucene.util.RamUsageEstimator;
import org.openjdk.jol.info.ClassLayout;

/**
 * @author bin
 * @since 2024/05/07
 */
public class ObjectSizeTest {
    public static void main() {
        // System.out.println(ClassLayout.parseInstance(new Object()).toPrintable());
        // System.out.println(ClassLayout.parseInstance(1L).toPrintable());
        System.out.println(ClassLayout.parseInstance(new A()).toPrintable());
        System.out.println(RamUsageEstimator.sizeOfObject(new A()));
        System.out.println(ClassLayout.parseInstance(new long[]{0, 0}).toPrintable());
        System.out.println(RamUsageEstimator.sizeOfObject(new long[]{0, 0}));
        System.out.println(ClassLayout.parseInstance(new B()).toPrintable());
        System.out.println(RamUsageEstimator.sizeOfObject(new B()));
        System.out.println(ClassLayout.parseInstance(new C()).toPrintable());
        System.out.println(RamUsageEstimator.sizeOfObject(new C()));
    }

    private static class A {
        private long a;
        private long b;
    }

    private static class B {
        private Long a;
        private Long b;
    }

    private static class C {
        private long[] a = new long[2];
    }

}
