package demo;

import lombok.RequiredArgsConstructor;
import lombok.val;
import org.intellij.lang.annotations.Language;

import java.util.Locale;

/**
 * @author bin
 * @since 2024/03/22
 */
@SuppressWarnings({"SqlNoDataSourceInspection"})
public class SqlSplit {
    private static final String SPLIT = "\t";
    @Language(value = "MySQL", prefix = "create table test(\n", suffix = "index)")
    private static final String STR = """
            """;

    private static void p(TableField field) {
        p(field.name);
        p(SPLIT);
        p(field.type);
        p(SPLIT);
        p(field.defaultValue);
        p(SPLIT);
        p(field.notNull ? "" : "null");
        p(SPLIT);
        p(field.comment);
        System.out.println();
    }

    public static void main() throws Exception {
        String tmp;
        // val list = new ArrayList<TableField>();
        for (String fieldStr : STR.split(",\n")) {
            TableField field = new TableField();
            val state = new State(fieldStr.toCharArray());
            state.length = state.cs.length;
            field.name = next(state);
            field.type = next(state);
            tmp = next(state);
            if (!tmp.isEmpty() && tmp.charAt(0) == '(') {
                field.type = field.type + " " + tmp;
            }
            do {
                switch (tmp) {
                    case "default" -> field.defaultValue = next(state);
                    case "auto_increment" -> field.increment = field.notNull = true;
                    case "not" -> {
                        next(state);
                        field.notNull = true;
                    }
                    case "null" -> field.notNull = false;
                    case "comment" -> {
                        tmp = next(state);
                        field.comment = tmp.substring(1, tmp.length() - 1);
                    }
                }
                tmp = next(state);
            } while (state.hasNext());
            p(field);
        }
    }

    private static void p(Object value) {
        if (value != null) {
            System.out.print(value);
        }
    }

    private static String next(State state) {
        while (state.hasNext() && Character.isWhitespace(state.getChar())) {
            state.next();
        }
        val builder = new StringBuilder();
        // ()
        int raw0 = 0;
        // ''
        boolean raw1 = false;
        // \
        boolean raw2 = false;
        while (state.hasNext()) {
            val c = state.getChar();
            if (raw0 == 0 && !raw1 && !raw2) {
                if (Character.isWhitespace(c)) {
                    break;
                }
            }
            builder.append(c);
            state.next();
            if (raw2) {
                raw2 = false;
                continue;
            }
            switch (c) {
                case '(' -> raw0++;
                case ')' -> raw0--;
                case '\'' -> raw1 = !raw1;
                case '\\' -> raw2 = true;
            }
        }
        return builder.toString().toLowerCase(Locale.ROOT);
    }

    @RequiredArgsConstructor
    private static final class State {
        public final char[] cs;
        public int length;
        public int index;

        public boolean hasNext() {
            return index < length;
        }

        public char getChar() {
            return cs[index];
        }

        public void next() {
            index++;
        }

        public void copy() {
            val index = this.index;
            val length = this.length - index;
            System.arraycopy(cs, index, cs, 0, length);
            this.length = length;
        }

        @Override
        public String toString() {
            return new String(cs);
        }
    }

    private static final class TableField {
        public String name;
        public String type;
        public String defaultValue;
        public boolean notNull;
        public boolean increment;
        public String comment;
    }
}
