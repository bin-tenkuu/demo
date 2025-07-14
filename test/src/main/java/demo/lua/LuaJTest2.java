package demo.lua;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import party.iroiro.luajava.JFunction;
import party.iroiro.luajava.Lua;
import party.iroiro.luajava.luajit.LuaJit;

import java.util.List;
import java.util.Map;

/**
 * {@link Lua#push} 之后需要使用 {@link Lua#setGlobal} 设置到全局变量中.
 * 如果如果 javaArray 需要让lua能修改最好使用 {@link Lua#pushJavaArray}.
 * @author bin
 * @since 2025/06/09
 */
@Slf4j
public class LuaJTest2 {
    public static void main(String[] args) {
        val start = System.currentTimeMillis();
        val params = Map.of(
                "arg", List.of("hash1", "hash2"),
                "ret", "hash3"
        );
        // language=lua
        val script = """
                local arg = params.arg
                local ret = params.ret
                local n = getData(arg[1]) + getData(arg[2])
                setData(ret, n)
                """;
        try (val L = new LuaJit()) {
            L.set("print", (JFunction) (l) -> {
                System.out.println(l.get().toString());
                return 0;
            });
            L.pushNil();
            L.setGlobal("java");
            set(L, "getData", (l) -> {
                val s = l.get().toString();
                val i = s.charAt(s.length() - 1) - '0';
                l.push(i);
                return 1;
            });
            set(L, "setData", (l) -> {
                val value = l.get().toNumber();
                val key = l.get().toString();
                log.info("{} => {}", key, value);
                return 0;
            });
            L.push(params);
            L.setGlobal("params");

            L.run(script);
        }

        System.out.println("Execution time: " + (System.currentTimeMillis() - start) + " ms");
    }

    private static void set(Lua l, String name, JFunction getter) {
        l.checkStack(1);
        l.push(getter);
        l.setGlobal(name);
    }

}
