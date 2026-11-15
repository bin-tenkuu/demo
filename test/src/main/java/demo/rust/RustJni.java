package demo.rust;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 调用 Rust JNI 的示例。
 *
 * <p>对应 rustdemo/src/lib.rs 中的 JNI 导出函数：
 * <pre>
 * Java_demo_rust_RustJni_addTwo__II
 * Java_demo_rust_RustJni_helloWorld__
 * Java_demo_rust_RustJni_setI__I
 * Java_demo_rust_RustJni_getI__
 * Java_demo_rust_RustJni_incrementI__
 * </pre>
 *
 * <p>运行前先在 rustdemo 目录执行 {@code cargo build}，生成
 * {@code rustdemo/target/debug/librustdemo.so}。
 *
 * @author bin
 * @since 2026/11/15
 */
public class RustJni {
    public static int i = 0;

    static {
        Path lib = Path.of("rustdemo", "target", "debug", "librustdemo.so")
                .toAbsolutePath();
        if (!Files.exists(lib)) {
            throw new UnsatisfiedLinkError("Rust 动态库不存在: " + lib + "，请先执行 cargo build");
        }
        System.load(lib.toString());
    }

    /**
     * 对应 Rust 的 add_two1，只是额外接收 JNIEnv* 和 jclass 两个参数。
     */
    public static native int addTwo(int a, int b);

    /**
     * 对应 Rust 的 hello_world。
     */
    public static native void helloWorld();

    /**
     * 把静态字段 {@link #i} 设置为指定值（由 Rust 写入）。
     */
    public static native void setI(int value);

    /**
     * 读取静态字段 {@link #i}（由 Rust 读取）。
     */
    public static native int getI();

    /**
     * 让 Rust 把 {@link #i} 加一，并返回新值。
     */
    public static native int incrementI();

    static void main() {
        helloWorld();
        int c = addTwo(1, 2);
        System.out.println("sum = " + c);

        System.out.println("before: i = " + i);
        setI(42);
        System.out.println("after setI(42): i = " + i);

        int next = incrementI();
        System.out.println("after incrementI(): i = " + i + ", returned = " + next);

        System.out.println("getI() = " + getI());
    }
}
