package demo.lua;

import lombok.val;
import org.luaj.vm2.Globals;
import org.luaj.vm2.LoadState;
import org.luaj.vm2.LuaTable;
import org.luaj.vm2.LuaValue;
import org.luaj.vm2.compiler.LuaC;
import org.luaj.vm2.lib.*;
import org.luaj.vm2.lib.jse.JseBaseLib;
import org.luaj.vm2.lib.jse.JseMathLib;

/**
 * @author bin
 * @since 2025/06/09
 */
public class LuaJTest {
    public static void main(String[] args) {
        val start = System.currentTimeMillis();

        // 加载自定义函数库
        Globals globals = new Globals();
        globals.load(new JseBaseLib());
        globals.load(new PackageLib());
        globals.load(new Bit32Lib());
        globals.load(new TableLib());
        globals.load(new StringLib());
        globals.load(new CoroutineLib());
        globals.load(new JseMathLib());
        LoadState.install(globals);
        LuaC.install(globals);
        val library = new ArgInsert(new String[]{"aaa", "bbb"});
        globals.load(library);

        globals.load("""
                print(arg[1])
                arg[1] = "hello"

                print(arg[1])
                print(arg[2])""").call();
        System.out.println(library.arg.rawget(1).checkstring());

        System.out.println("Execution time: " + (System.currentTimeMillis() - start) + " ms");
    }

    private static class ArgInsert extends TwoArgFunction {
        private final LuaTable arg = tableOf();

        public ArgInsert(String[] args) {
            for (int i = 0, argsLength = args.length; i < argsLength; i++) {
                var s = args[i];
                arg.rawset(i + 1, valueOf(s));
            }
        }

        @Override
        public LuaValue call(LuaValue modname, LuaValue env) {
            env.set("arg", arg);
            env.get("package").get("loaded").set("arg", arg);
            return arg;
        }
    }
}
