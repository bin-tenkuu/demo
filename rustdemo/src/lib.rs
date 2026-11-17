//! 文档：https://kaisery.github.io/trpl-zh-cn/title-page.html
pub mod jni_binding;
pub mod learn;

#[unsafe(no_mangle)]
pub extern "C" fn add_two1(a: u32, b: u32) -> u32 {
    let c = a + b;
    println!("print in rust, sum is: {}", c);
    c
}

#[unsafe(no_mangle)]
pub extern "C" fn hello_world() {
    println!("Hello, world from Rust!");
}

pub use jni_binding::RustJni;
