package demo;

import org.graalvm.polyglot.Context;

/**
 * @author bin
 * @since 2025/12/12
 */
public class PolyglotTest {
    static void main() {
        try (var polyglot = Context.newBuilder()
                .allowAllAccess(true)
                .build()) {
            for (var language : polyglot.getEngine().getLanguages().keySet()) {
                System.out.println(language);
            }
            var js = polyglot.getBindings("js");
            js.putMember("a", new int[]{1, 2, 3, 4});
            // language=js
            var array = polyglot.eval("js", """
                    // let a = [1,42,3,4]
                    console.log("a", a)
                    console.log("a[2]", a[2])
                    a
                    """);
            var result = array.getArrayElement(2);
            System.out.println(result.asInt());
            var func = polyglot.eval("js", "x=>x+1");
            result = func.execute(result);
            System.out.println(result.asInt());
            // language=js
            result = polyglot.eval("js", """
                    const BigDecimal = Java.type('java.math.BigDecimal');
                    BigDecimal.valueOf(10).pow(20)""");
            System.out.println(result);
        }
    }
}
