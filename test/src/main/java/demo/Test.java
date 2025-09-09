package demo;

import java.lang.reflect.Field;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/03/04
 */
public class Test {
    private static final Charset GBK = Charset.forName("GBK");
    private static final Field value;

    static {
        try {
            value = String.class.getDeclaredField("value");
            // value.setAccessible(true);
        } catch (NoSuchFieldException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) throws IllegalAccessException {
        var s = new String("测试".getBytes(GBK), StandardCharsets.UTF_8);
        System.out.println(s);
        var bytes = (byte[]) value.get(s);
        var gbk = new String(bytes, GBK);
        System.out.println(gbk);
    }

}
