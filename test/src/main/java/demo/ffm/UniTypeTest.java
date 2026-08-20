package demo.ffm;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.UnionLayout;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

import static java.lang.foreign.ValueLayout.*;

/**
 * @author bin
 * @since 2026/08/20
 */
public class UniTypeTest {
    private static final ByteOrder BYTE_ORDER = ByteOrder.BIG_ENDIAN;
    private static final UnionLayout RECORD_LAYOUT = MemoryLayout.unionLayout(
            JAVA_LONG_UNALIGNED.withOrder(BYTE_ORDER).withName("recordId"),
            MemoryLayout.structLayout(
                    JAVA_SHORT_UNALIGNED.withOrder(BYTE_ORDER).withName("tableId"),
                    JAVA_INT_UNALIGNED.withOrder(BYTE_ORDER).withName("serNo"),
                    JAVA_SHORT_UNALIGNED.withOrder(BYTE_ORDER).withName("columnNo")
            ).withName("parts")
    ).withName("record");

    private static final VarHandle RECORD_ID_HANDLE = RECORD_LAYOUT.varHandle(
            PathElement.groupElement("recordId")
    );
    private static final VarHandle TABLE_ID_HANDLE = RECORD_LAYOUT.varHandle(
            PathElement.groupElement("parts"),
            PathElement.groupElement("tableId")
    );
    private static final VarHandle SER_NO_HANDLE = RECORD_LAYOUT.varHandle(
            PathElement.groupElement("parts"),
            PathElement.groupElement("serNo")
    );
    private static final VarHandle COLUMN_NO_HANDLE = RECORD_LAYOUT.varHandle(
            PathElement.groupElement("parts"),
            PathElement.groupElement("columnNo")
    );

    public static void main() {
        writeRecordIdThenReadParts();
        writePartsThenReadRecordId();
    }

    private static void writeRecordIdThenReadParts() {
        long recordId = pack((short) 1100, 1, (short) 16);
        try (var arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(RECORD_LAYOUT);
            RECORD_ID_HANDLE.set(segment, 0L, recordId);
            System.out.println("write recordId:");
            System.out.println("recordId=" + recordId);
            System.out.println("tableId=" + tableIdOf(segment));
            System.out.println("serNo=" + serNoOf(segment));
            System.out.println("columnNo=" + columnNoOf(segment));
            System.out.println();
        }
    }

    private static void writePartsThenReadRecordId() {
        short tableId = (short) 1100;
        int serNo = 1;
        short columnNo = (short) 16;
        try (var arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(RECORD_LAYOUT);
            TABLE_ID_HANDLE.set(segment, 0L, (short) tableId);
            SER_NO_HANDLE.set(segment, 0L, (int) serNo);
            COLUMN_NO_HANDLE.set(segment, 0L, (short) columnNo);
            long recordId = (long) RECORD_ID_HANDLE.get(segment, 0L);
            System.out.println("write parts:");
            System.out.println("tableId=" + tableId);
            System.out.println("serNo=" + serNo);
            System.out.println("columnNo=" + columnNo);
            System.out.println("recordId=" + recordId);
            System.out.println("pack(tableId, serNo, columnNo)=" + pack(tableId, serNo, columnNo));
            System.out.println("unpack.tableId=" + tableIdOf(recordId));
            System.out.println("unpack.serNo=" + serNoOf(recordId));
            System.out.println("unpack.columnNo=" + columnNoOf(recordId));
            System.out.println();
        }
    }

    private static long pack(short tableId, int serNo, short columnNo) {
        return (long) tableId << 48
                | (long) serNo << 16
                | columnNo;
    }

    private static short tableIdOf(long recordId) {
        return (short) (recordId >>> 48 & 0xFFFFL);
    }

    private static int serNoOf(long recordId) {
        return (int) (recordId >>> 16 & 0xFFFF_FFFFL);
    }

    private static short columnNoOf(long recordId) {
        return (short) (recordId & 0xFFFFL);
    }

    private static short tableIdOf(MemorySegment segment) {
        return (short) TABLE_ID_HANDLE.get(segment, 0L);
    }

    private static int serNoOf(MemorySegment segment) {
        return (int) SER_NO_HANDLE.get(segment, 0L);
    }

    private static short columnNoOf(MemorySegment segment) {
        return (short) COLUMN_NO_HANDLE.get(segment, 0L);
    }

}
