package demo;

import java.lang.reflect.Proxy;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/08/15
 */
public class ProxyTest {
    public static void main() {
        var instance = (I) Proxy.newProxyInstance(ProxyTest.class.getClassLoader(), new Class[]{I.class},
                (proxy, method, args) -> {
                    System.out.println(method.getName());
                    switch (method.getName()) {
                        case "testInt", "testString" -> {
                            return args[0];
                        }
                        default -> {
                            return null;
                        }
                    }
                }
        );
        instance.test();
        System.out.println(instance.testInt(1));
        System.out.println(instance.testString("test"));
    }

    private interface I {
        void test();

        int testInt(int a);

        String testString(String a);
    }
}
