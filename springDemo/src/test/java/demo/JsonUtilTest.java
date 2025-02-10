package demo;

import demo.util.JsonUtil;
import lombok.val;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/02/10
 */
@SpringBootTest
public class JsonUtilTest {
    @Test
    public void test() {
        val v1 = JsonUtil.tryParse(om -> om.readValue("\"1\"", Long.class));
        System.out.println(v1);
        val v2 = JsonUtil.tryParse("\"1\"", (om, t) -> om.readValue(t, Long.class));
        System.out.println(v2);
    }
}
