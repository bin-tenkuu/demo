package demo.ffm;

import tools.jackson.databind.json.JsonMapper;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Random;

/**
 * @author bin
 * @since 2026/02/10
 */
public class MMapTest {
    private static final Path SHARED_FILE = Path.of("./mmap.tmp");
    private static final int SIZE = 1 << 20; // 1M
    private static final JsonMapper json = new JsonMapper();

    public static class Write {
        static void main() throws IOException, InterruptedException {
            // 确保文件存在且大小足够
            if (!Files.isRegularFile(SHARED_FILE)) {
                Files.createFile(SHARED_FILE);
            }
            // 设置文件大小（重要！）
            try (var raf = new RandomAccessFile(SHARED_FILE.toFile(), "rw")) {
                raf.setLength(SIZE);
            }
            try (var arena = Arena.ofConfined();
                 var channel = FileChannel.open(SHARED_FILE, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
                var segment = channel.map(FileChannel.MapMode.READ_WRITE, 0, SIZE, arena);
                var random = new Random();
                var list = new ArrayList<Integer>();
                while (true) {
                    if (list.size() < 10000) {
                        for (var i = 0; i < 10; i++) {
                            list.add(random.nextInt());
                        }
                    } else {
                        System.err.println("full");
                        Thread.sleep(1);
                    }
                    var size = segment.get(ValueLayout.JAVA_INT, 0);
                    if (size != 0) {
                        // 长度不为 0 表示没有数据，等待消费后再写入
                        System.out.println("wait");
                        Thread.sleep(1);
                    } else {
                        var str = json.writeValueAsBytes(list);
                        list.clear();
                        size = str.length;
                        var slice = segment.asSlice(ValueLayout.JAVA_INT.byteSize(), size);
                        slice.copyFrom(MemorySegment.ofArray(str));
                        // 先写入数据，再设置长度，确保消费者读取到完整数据
                        segment.set(ValueLayout.JAVA_INT, 0, size);
                        System.out.println("write");
                    }
                }
            }
        }
    }

    public static class Read {
        static void main() throws IOException, InterruptedException {
            // 确保文件存在且大小足够
            if (!Files.isRegularFile(SHARED_FILE)) {
                System.err.println("Shared file does not exist.");
                return;
            }
            try (var arena = Arena.ofConfined();
                 var channel = FileChannel.open(SHARED_FILE, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
                var segment = channel.map(FileChannel.MapMode.READ_WRITE, 0, SIZE, arena);
                while (true) {
                    var size = segment.get(ValueLayout.JAVA_INT, 0);
                    if (size == 0) {
                        // 长度为 0 表示没有数据，等待生产后再读取
                        System.out.println("wait");
                        Thread.sleep(1);
                    } else {
                        var slice = segment.asSlice(ValueLayout.JAVA_INT.byteSize(), size);
                        var str = slice.toArray(ValueLayout.JAVA_BYTE);
                        // 先读取数据，再设置长度为 0
                        segment.set(ValueLayout.JAVA_INT, 0, 0);
                        System.out.print("read: ");
                        System.out.println(new String(str));
                    }
                }
            }
        }
    }
}
