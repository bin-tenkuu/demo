package demo;

public class ScopedValueTest {
    private static final ScopedValue<String> X = ScopedValue.newInstance();

    private static void foo() {
        ScopedValue.where(X, "hello").run(ScopedValueTest::bar);
    }

    private static void bar() {
        System.out.println(X.get()); // prints hello
        ScopedValue.where(X, "goodbye").run(ScopedValueTest::baz);
        System.out.println(X.get()); // prints hello
    }

    private static void baz() {
        System.out.println(X.get()); // prints goodbye
    }

    public static void main(String[] args) {
        foo();
    }
}
