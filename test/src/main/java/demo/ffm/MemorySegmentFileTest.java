package demo.ffm;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Formatter;

/**
 * @author bin
 * @since 2026/01/26
 */
public class MemorySegmentFileTest {
    private static final Path path = Path.of("/home/bin-/Downloads/home/extraData.img");

    static void main() {
        try (var arena = Arena.ofConfined();
             var channel = FileChannel.open(path, StandardOpenOption.READ)) {
            var size = channel.size();
            var memorySegment = channel.map(FileChannel.MapMode.READ_ONLY,
                    0,
                    size,
                    arena
            );

            System.out.println("成功映射文件，大小：" + printSize(size));
            System.out.println("内存段地址：" + memorySegment.address());
            System.out.println("内存段大小：" + printSize(memorySegment.byteSize()));
            System.out.println("内存段内容前16字节：");
            for (long i = 0; i < 16; i++) {
                byte b = memorySegment.get(ValueLayout.JAVA_BYTE, i);
                System.out.printf("0x%02X ", b);
            }
            System.out.println();
            System.out.println("内存段内容后16字节：");
            for (long i = size - 16; i < size; i++) {
                byte b = memorySegment.get(ValueLayout.JAVA_BYTE, i);
                System.out.printf("0x%02X ", b);
            }
            System.out.println();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static String printSize(long size) {
        String[] units = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
        long s = size;
        var list = new ArrayList<String>();
        var builder = new StringBuilder();
        var formatter = new Formatter(builder);
        for (var unit : units) {
            int sub = (int) (s & ((1 << 10) - 1));
            formatter.format("%s %s", sub, unit);
            list.add(builder.toString());
            builder.setLength(0);
            if (s < 1024) {
                break;
            }
            s >>= 10;
        }
        return String.join(" ", list.reversed());
    }
}
