package demo;

import lombok.val;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.VarHandle;
import java.util.Arrays;

import static java.lang.foreign.ValueLayout.*;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/07/21
 */
public class FFMTest {
    private static final Linker linker = Linker.nativeLinker();
    private static final SymbolLookup lookup = linker.defaultLookup();
    private static final Arena arena = Arena.ofAuto();

    public static void main() throws Throwable {
        stringTest();
        strlen();
        qsort();
        struct();
    }

    private static void struct() throws Throwable {
        // struct Point {
        //     int x;
        //     int y;
        // } pts[10];
        StructLayout pointLayout = MemoryLayout.structLayout(
                JAVA_INT.withName("x"),
                JAVA_INT.withName("y")
        );
        SequenceLayout pointsLayout = MemoryLayout.sequenceLayout(10,
                pointLayout
        );
        MethodHandle sliceHandle = pointsLayout.sliceHandle(PathElement.sequenceElement());
        VarHandle xHandle = pointsLayout.varHandle(PathElement.sequenceElement(),
                PathElement.groupElement("x"));
        VarHandle yHandle = pointsLayout.varHandle(PathElement.sequenceElement(),
                PathElement.groupElement("y"));
        MemorySegment segment = arena.allocate(pointsLayout);
        for (int i = 0; i < pointsLayout.elementCount(); i++) {
            xHandle.set(segment,
                    /* base */ 0L,
                    /* index */ (long) i,
                    /* value to write */ 10 + i); // x
            yHandle.set(segment,
                    /* base */ 0L,
                    /* index */ (long) i,
                    /* value to write */ 100 + i); // y
        }
        VarHandle xPHandle = pointLayout.varHandle(PathElement.groupElement("x"));
        VarHandle yPHandle = pointLayout.varHandle(PathElement.groupElement("y"));
        for (int i = 0; i < pointsLayout.elementCount(); i++) {
            val point = (MemorySegment) sliceHandle.invoke(segment, 0L, (long) i);
            System.out.printf("index[%d].x=%s,%s%n",
                    i, xHandle.get(segment, 0L, i), xPHandle.get(point, 0));
            System.out.printf("index[%d].y=%s,%s%n",
                    i, yHandle.get(segment, 0L, i), yPHandle.get(point, 0));
        }
    }

    private static void qsort() throws Throwable {
        // MethodHandles.lookup().unreflect(FFMTest.class.getDeclaredMethod("qsortCompare", MemorySegment.class, MemorySegment.class));
        MethodHandle comparHandle = MethodHandles.lookup().findStatic(FFMTest.class, "qsortCompare",
                MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class)
        );
        MemorySegment comparFunc = linker.upcallStub(comparHandle,
                FunctionDescriptor.of(JAVA_INT,
                        ADDRESS.withTargetLayout(JAVA_INT),
                        ADDRESS.withTargetLayout(JAVA_INT)),
                arena
        );
        // void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
        MethodHandle qsort = linker.downcallHandle(
                lookup.find("qsort").get(),
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS)
        );
        MemorySegment array = arena.allocateFrom(JAVA_INT,
                0, 9, 3, 4, 6, 5, 1, 8, 2, 7
        );
        qsort.invoke(array, 10L, JAVA_INT.byteSize(), comparFunc);
        int[] sorted = array.toArray(JAVA_INT);
        System.out.println(Arrays.toString(sorted)); // [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
    }

    private static int qsortCompare(MemorySegment elem1, MemorySegment elem2) {
        return Integer.compare(elem1.get(JAVA_INT, 0), elem2.get(JAVA_INT, 0));
    }

    private static void strlen() throws Throwable {
        // size_t strlen(const char *s);
        MethodHandle strlen = linker.downcallHandle(
                lookup.find("strlen").get(),
                FunctionDescriptor.of(JAVA_LONG, ADDRESS)
        );
        MemorySegment str = arena.allocateFrom("Hello");
        long len = (long) strlen.invoke(str);
        System.out.println(len); // 5
    }

    private static void stringTest() {
        MemorySegment cString = arena.allocateFrom("Panama");
        String jString = cString.getString(1L);
        System.out.println(jString);
    }

}
