package demo.lua;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import party.iroiro.luajava.JFunction;
import party.iroiro.luajava.Lua;
import party.iroiro.luajava.luajit.LuaJit;

import java.util.HashMap;
import java.util.Map;

/**
 * @author bin
 * @since 2025/06/09
 */
@Slf4j
public class LuaJTest {
    public static void main(String[] args) {
        val start = System.currentTimeMillis();

        val value = new String[]{"aaa", "bbb"};
        try (val L = new LuaJit()) {
            L.set("print", (JFunction) (l) -> {
                System.out.println(l.get().toString());
                return 0;
            });
            L.set("log", log);
            L.set("arg", value);
            // language=lua
            L.run("""
                    log:info("{}, {}", arg[1], arg[2])
                    arg[1] = "hello"
                    log:info("{}, {}", arg[1], arg[2])
                    """);
            System.out.println(L.get("_VERSION"));
        }
        System.out.println("===");
        System.out.println(value[0]);

        System.out.println("Execution time: " + (System.currentTimeMillis() - start) + " ms");
    }

    public static Map<String, JFunction> logTable() {
        val map = new HashMap<String, JFunction>();
        map.put("info", L -> {
            val objects = new Object[L.getTop() - 1];
            val str = L.get();
            for (var i = 0; i < objects.length; i++) {
                objects[i] = L.get();
            }
            log.info(str.toString(), objects);
            return 0;
        });
        map.put("debug", L -> {
            val objects = new Object[L.getTop() - 1];
            val str = L.get();
            for (var i = 0; i < objects.length; i++) {
                objects[i] = L.get();
            }
            log.debug(str.toString(), objects);
            return 0;
        });
        map.put("warn", L -> {
            val objects = new Object[L.getTop() - 1];
            val str = L.get();
            for (var i = 0; i < objects.length; i++) {
                objects[i] = L.get();
            }
            log.warn(str.toString(), objects);
            return 0;
        });
        map.put("error", L -> {
            val objects = new Object[L.getTop() - 1];
            val str = L.get();
            for (var i = 0; i < objects.length; i++) {
                objects[i] = L.get();
            }
            log.error(str.toString(), objects);
            return 0;
        });
        map.put("trace", L -> {
            val objects = new Object[L.getTop() - 1];
            val str = L.get();
            for (var i = 0; i < objects.length; i++) {
                objects[i] = L.get();
            }
            log.trace(str.toString(), objects);
            return 0;
        });
        map.put("print", L -> {
            for (int i = 1; i <= L.getTop(); i++) {
                System.out.print(L.get().toString());
            }
            return 0;
        });
        return map;
    }

    private interface Arg0JFunction extends JFunction {
        @Override
        default int __call(party.iroiro.luajava.Lua L) {
            invoke(L);
            return 0;
        }

        void invoke(Lua L);
    }
}
