package demo.lua;

import party.iroiro.luajava.JFunction;
import party.iroiro.luajava.Lua;
import party.iroiro.luajava.value.LuaFunction;
import party.iroiro.luajava.value.LuaValue;

import java.lang.reflect.Array;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * @author bin
 * @since 2025/08/19
 */
@SuppressWarnings("unused")
public class LuaUtil {
    private static final HashMap<Class<?>, BiConsumer<Lua, ?>> pushMap = new HashMap<>();

    public static void clearPush() {
        pushMap.clear();
    }

    public static <T> void registerPush(Class<T> clazz, BiConsumer<Lua, T> consumer) {
        pushMap.put(clazz, consumer);
    }

    @SuppressWarnings("unchecked")
    public static <T> void push(Lua L, Class<?> clazz, T o) {
        var push = (BiConsumer<Lua, Object>) pushMap.get(clazz);
        if (push != null) {
            push.accept(L, o);
        } else {
            throw new IllegalArgumentException("No push function registered for class: " + clazz.getName());
        }

    }

    public static void push(Lua L) {
        L.pushNil();
    }

    public static void push(Lua L, LuaValue luaValue) {
        L.push(luaValue);
    }

    public static void push(Lua L, LuaFunction luaFunction) {
        L.push(luaFunction);
    }

    public static void push(Lua L, boolean b) {
        L.push(b);
    }

    public static void push(Lua L, long l) {
        L.push(l);
    }

    public static void push(Lua L, Number number) {
        L.push(number);
    }

    public static void push(Lua L, String string) {
        L.push(string);
    }

    public static void push(Lua L, JFunction jFunction) {
        L.push(jFunction);
    }

    public static void push(Lua L, Map<?, ?> map) {
        L.checkStack(3);
        L.createTable(0, map.size());
        for (var entry : map.entrySet()) {
            push(L, entry.getKey());
            push(L, entry.getValue());
            L.rawSet(-3);
        }
    }

    public static void push(Lua L, Collection<?> os) {
        L.checkStack(2);
        L.createTable(os.size(), 0);
        int i = 1;
        for (Object o : os) {
            push(L, o);
            L.rawSetI(-2, i);
            i++;
        }
    }

    public static void pushArray(Lua L, Object array) {
        int len = Array.getLength(array);
        L.createTable(len, 0);
        int i = 0;
        while (i < len) {
            push(L, Array.get(array, i));
            i++;
            L.rawSetI(-2, i);
        }
    }

    public static void push(Lua L, Object o) {
        if (o == null) {
            push(L);
            return;
        }
        var clazz = o.getClass();
        if (clazz.isArray()) {
            pushArray(L, o);
            return;
        }
        switch (o) {
            case LuaValue luaValue -> push(L, luaValue);
            case LuaFunction luaFunction -> push(L, luaFunction);
            case Boolean b -> push(L, b.booleanValue());
            case Byte b -> push(L, b.longValue());
            case Short s -> push(L, s.longValue());
            case Integer i -> push(L, i.longValue());
            case Character c -> push(L, c.charValue());
            case Long l -> push(L, l.longValue());
            case Float f -> push(L, f);
            case Double d -> push(L, d);
            case Number n -> push(L, n);
            case String s -> push(L, s);
            case JFunction f -> push(L, f);
            case Map<?, ?> map -> push(L, map);
            case Collection<?> c -> push(L, c);
            default -> push(L, clazz, o);
        }
    }

    public static void set(Lua L, String name, Object o) {
        push(L, o);
        L.setGlobal(name);
    }

}
