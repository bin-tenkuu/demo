import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/03/04
 */
public class Test {
    private static final String prefix = "";
    private static final File fromFile = new File("./files.txt");
    private static final File toDir = new File("./to/");

    public static void main(String[] args) throws IOException {
        try (var reader = Files.newBufferedReader(fromFile.toPath())) {
            String line;
            while ((line = reader.readLine()) != null) {
                var file = new File(line);
                if (file.exists() && file.isFile()) {
                    var subPath = line.substring(prefix.length());
                    var toFile = new File(toDir, subPath);
                    toFile.getParentFile().mkdirs();
                    Files.copy(file.toPath(), toFile.toPath());
                }
            }
        }
    }

}
