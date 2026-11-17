//! ===== JNI 示例 =====
//! 对应 Java 类：test/src/main/java/demo/rust/RustJni.java
//!
//! 用 bind_java_type! 一次性生成整个绑定：
//!   - RustJni<'local>：Java 对象的 Rust 包装类型（Reference + Deref<JObject>）
//!   - RustJniAPI：缓存 class 引用和字段/方法 ID，并负责 RegisterNatives
//!   - RustJniNativeInterface：native 方法的实现 trait
//!   - 每个 native 方法的 JNI 导出符号（native_methods_export 默认 true）
//!   - 静态字段 i 的访问器：RustJni::i(env) / RustJni::set_i(env, value)
//!
//! native_methods 的命名规则与 native_method! 一致：Rust snake_case 自动转
//! lowerCamelCase，所以 add_two / hello_world / set_i / get_i / increment_i
//! 正好对应 Java 的 addTwo / helloWorld / setI / getI / incrementI。
// bind_java_type! 生成的 ABI 校验用了新版 Rust 已改名的
// AtomicBool::fetch_update。用模块级 allow 收口，避免屏蔽 crate 里其他 deprecated 警告。
#![allow(deprecated)]

use jni::objects::{JClass, JValue};
use jni::sys::jint;
use jni::{Env, bind_java_type, jni_sig, jni_str};

bind_java_type! {
    pub RustJni => demo.rust.RustJni,

    fields {
        /// Java: `public static int i`
        static i: jint,
    },

    native_methods {
        /// Java: `public static native int addTwo(int a, int b)`
        // static extern fn add_two(a: jint, b: jint) -> jint,
        static extern fn add_two {
            name = "addTwo",
            sig = (a: jint, b: jint) -> jint,
        },

        /// Java: `public static native void helloWorld()`
        // static extern fn hello_world(),
        static extern fn hello_world {
            name = "helloWorld",
            sig = (),
        },

        /// Java: `public static native void setI(int value)`
        static extern fn set_i(value: jint),

        /// Java: `public static native int getI()`
        static extern fn get_i() -> jint,

        /// Java: `public static native int incrementI()`
        static extern fn increment_i() -> jint,
    },
}

impl RustJniNativeInterface for RustJniAPI {
    type Error = jni::errors::Error;

    fn add_two<'local>(
        _env: &mut Env<'local>,
        _class: JClass<'local>,
        a: jint,
        b: jint,
    ) -> Result<jint, Self::Error> {
        let c = a + b;
        println!("print in rust(jni), sum is: {}", c);
        Ok(c)
    }

    fn hello_world<'local>(
        _env: &mut Env<'local>,
        _class: JClass<'local>,
    ) -> Result<(), Self::Error> {
        println!("Hello, world from Rust JNI!");
        Ok(())
    }

    /// 把静态字段 `i` 设置为 `value`。
    fn set_i<'local>(
        env: &mut Env<'local>,
        class: JClass<'local>,
        value: jint,
    ) -> Result<(), Self::Error> {
        env.set_static_field(&class, jni_str!("i"), jni_sig!("I"), JValue::Int(value))?;
        Ok(())
    }

    /// 读取静态字段 `i`。
    fn get_i<'local>(env: &mut Env<'local>, class: JClass<'local>) -> Result<jint, Self::Error> {
        Ok(env
            .get_static_field(&class, jni_str!("i"), jni_sig!("I"))?
            .i()?)
    }

    /// 读取 `i`，加一后写回，并返回新值。
    fn increment_i<'local>(
        env: &mut Env<'local>,
        class: JClass<'local>,
    ) -> Result<jint, Self::Error> {
        let current = env
            .get_static_field(&class, jni_str!("i"), jni_sig!("I"))?
            .i()?;
        let next = current + 1;

        env.set_static_field(&class, jni_str!("i"), jni_sig!("I"), JValue::Int(next))?;

        Ok(next)
    }
}
