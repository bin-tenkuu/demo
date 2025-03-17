package demo;

import lombok.val;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.TreeSet;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/03/04
 */
public class Test {
    private static final Charset GBK = Charset.forName("GBK");

    public static void main(String[] args) throws IOException {
        val file = new File("OperationMode1_kxdw(1).dat");
        val treeSet = new TreeSet<String>();
        val list = new ArrayList<Line>();
        try (val reader = Files.newBufferedReader(file.toPath(), GBK)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.length() < 67 || !line.startsWith("L ")) {
                    continue;
                }
                val st1 = line.substring(6, 17);
                val st2 = line.substring(19, 30);
                val lineId = line.substring(66).trim();
                treeSet.add(st1);
                treeSet.add(st2);
                val line1 = new Line(lineId, st1, st2);
                list.add(line1);
            }
        }
        val sb = new StringBuilder();
        sb.append("{");
        sb.append("\"sts\":[");
        for (String st : treeSet) {
            sb.append("\"").append(st).append("\",");
        }
        sb.setLength(sb.length() - 1);
        sb.append("],");
        sb.append("\"lines\":[");
        for (Line line : list) {
            sb.append(line).append(",");
        }
        sb.setLength(sb.length() - 1);
        sb.append("]");
        sb.append("}");
        Files.writeString(new File("OperationMode1_kxdw(1).json").toPath(), sb);
    }

    private record Line(String lineId, String st1, String st2) {
        @Override
        public String toString() {
            return "{\"lineId\":\"" + lineId + "\",\"st1\":\"" + st1 + "\",\"st2\":\"" + st2+"\"}";
        }
    }
}
