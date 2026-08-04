package demo;

import java.io.File;

/**
 * @author bin
 * @since 2026/08/04
 */
public class DirNameTest {
    static void main() {
        var base = new File("/home/bin-/Downloads/华北电碳协同截图");
        rename(base, base, "");
    }

    private static void rename(File base, File dir, String prefix) {
        var files = dir.listFiles();
        if (files == null) {
            return;
        }
        for (var file : files) {
            if (file.isDirectory()) {
                var name = file.getName();
                if (!prefix.isEmpty()) {
                    name = prefix + "-" + name;
                }
                rename(base, file, name);
            }
            if (file.isFile()) {
                var name = file.getName();
                var extension = name.substring(name.lastIndexOf('.'));
                file.renameTo(new File(base, prefix + extension));
            }
        }

    }
}
