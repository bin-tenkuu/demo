package demo;

import demo.util.PorxyClassUtil;
import lombok.val;

import java.io.IOException;
import java.lang.invoke.MethodHandles;

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

    public static void main(String[] args) throws Throwable {
        val util = new PorxyClassUtil(true);
        // util.helloworld();
        A a = util.proxy(A.class);
        a.test();
        a.test(1);
        a = PorxyClassUtil.proxyLookup(A.class);
        a.test();
        a.test(1);
    }

}
