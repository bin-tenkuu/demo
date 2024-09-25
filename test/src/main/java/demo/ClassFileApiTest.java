package demo;

import demo.util.PorxyClassUtil;
import lombok.val;

import java.io.IOException;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/25
 */
public class ClassFileApiTest {

    public static class A {
        public void test() {
            System.out.println("Hello, World!");
        }

        public int test(int a) {
            return a;
        }
    }

    public static void main(String[] args) throws IOException {
        val a = new PorxyClassUtil(true).proxy(A.class);
        a.test();
        a.test(1);
    }
}
