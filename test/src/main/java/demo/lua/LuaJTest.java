package demo.lua;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import party.iroiro.luajava.JFunction;
import party.iroiro.luajava.Lua;
import party.iroiro.luajava.luajit.LuaJit;
import party.iroiro.luajava.value.ImmutableLuaValue;
import party.iroiro.luajava.value.LuaTableValue;
import party.iroiro.luajava.value.LuaValue;
import party.iroiro.luajava.value.RefLuaValue;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * {@link Lua#push} 之后需要使用 {@link Lua#setGlobal} 设置到全局变量中.
 * 如果如果 javaArray 需要让lua能修改最好使用 {@link Lua#pushJavaArray}.
 * @author bin
 * @since 2025/06/09
 */
@Slf4j
public class LuaJTest {
    public static void main(String[] args) {
        val start = System.currentTimeMillis();
        val arg1 = new String[]{"aaa", "bbb"};
        val arg2 = new String[]{"aaa", "bbb"};
        try (val L = new LuaJit()) {
            L.set("print", (JFunction) (l) -> {
                System.out.println(l.get().toString());
                return 0;
            });
            L.pushNil();
            L.setGlobal("java");
            L.push(logTable());
            L.setGlobal("log");
            L.pushArray(arg1);
            L.setGlobal("arg1");
            L.pushJavaArray(arg2);
            L.setGlobal("arg2");
            // language=lua
            L.run("""
                    log.info("arg1: {}, {}", arg1[1], arg1[2])
                    arg1[1] = "hello"
                    log.info("arg1: {}, {}", arg1[1], arg1[2])
                    log.info("arg2: {}, {}", arg2[1], arg2[2])
                    arg2[1] = "hello"
                    log.info("arg2: {}, {}", arg2[1], arg2[2])
                    """);
            System.out.println(arg1[0]);
            System.out.println(arg2[0]);
            System.out.println("===");
            log.info("print global table");
            print("", L.get("_G"));
        }

        System.out.println("Execution time: " + (System.currentTimeMillis() - start) + " ms");
    }

    private static void print(String tab, LuaValue v) {
        switch (v) {
            case ImmutableLuaValue<?> immutable -> {
                System.out.print(immutable);
            }
            case RefLuaValue ref -> {
                val type = ref.type();
                System.out.print("<");
                System.out.print(type);
                System.out.print("> ");
                if (type != Lua.LuaType.FUNCTION) {
                    System.out.print(ref);
                }
            }
            case LuaTableValue table -> {
                System.out.print("<map> ");
                for (var entry : table.entrySet()) {
                    System.out.println();
                    val key = entry.getKey();
                    System.out.print(tab);
                    print(tab + "\t", key);
                    System.out.print("\t=>\t");
                    if (key.toString().equals("_G")) {
                        System.out.print("<_G>");
                    } else {
                        print(tab + "\t", entry.getValue());
                    }
                }
            }
            default -> {
                System.out.print("<");
                System.out.print(v.getClass().getName());
                System.out.print("> ");
                System.out.println();
            }
        }
    }

    private static JFunction toJFunction(BiConsumer<String, Object[]> logger) {
        return L -> {
            val size = L.getTop();
            val objects = new Object[size - 1];
            for (var i = size - 2; i >= 0; i--) {
                objects[i] = L.get().toString();
            }
            val str = L.get().toString();
            logger.accept(str, objects);
            return 0;
        };
    }

    public static Map<String, JFunction> logTable() {
        val map = new HashMap<String, JFunction>();
        map.put("info", toJFunction(log::info));
        map.put("debug", toJFunction(log::debug));
        map.put("warn", toJFunction(log::warn));
        map.put("error", toJFunction(log::error));
        map.put("trace", toJFunction(log::trace));
        return map;
    }

}
