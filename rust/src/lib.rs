#[unsafe(no_mangle)]
pub extern "C" fn addtwo1(a: u32, b: u32) -> u32 {
    let c = a + b;
    println!("print in rust, sum is: {}", c);
    c
}

#[unsafe(no_mangle)]
pub extern "C" fn hello_world() {
    println!("Hello, world from Rust!");
}
