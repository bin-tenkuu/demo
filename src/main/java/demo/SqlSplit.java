package demo;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author bin
 * @since 2024/03/22
 */
public class SqlSplit {
    private static final String SPLIT = "\t";
    private static final Pattern pattern = Pattern.compile(
            "(?<name>[^ ]+)\\s+" +
            "(?<type>[^ ]+)\\s+" +
            "(?:default (?<default>[^ ]+)\\s+)?" +
            "(?<increment>auto_increment\\s+)?" +
            "(?:(?<nonull>not )?null\\s+)?" +
            "(?:comment '(?<comment>[^']+)')?",
            Pattern.CASE_INSENSITIVE
    );
    private static final String STR = """
            """;

    public static void main() {
        for (String s : STR.split(",\n")) {
            Matcher matcher = pattern.matcher(s);
            Handler handler = new Handler(matcher);
            if (matcher.find()) {
                handler.printFirst("name");
                handler.print("type");
                handler.print("default");
                handler.printNotNull("nonull", "", "null");
                handler.print("comment");
                handler.end();
            }
        }
    }

    private record Handler(Matcher matcher) {
        private void p(Object value) {
            if (value != null) {
                System.out.print(value);
            }
        }

        public void printFirst(String key) {
            p(matcher.group(key));
        }

        public void print(String key) {
            p(SPLIT);
            p(matcher.group(key));
        }

        public void printNotNull(String key, Object pass, Object fail) {
            p(SPLIT);
            String s = matcher.group(key);
            if (s != null) {
                p(pass);
            } else {
                p(fail);
            }
        }

        public void end() {
            System.out.println();
        }
    }

}
