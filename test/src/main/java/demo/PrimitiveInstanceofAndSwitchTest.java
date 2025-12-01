package demo;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/24
 */
public class PrimitiveInstanceofAndSwitchTest {
    public static void main(String[] args) {
        switchTest(0);
        switchTest(1);
        switchTest(-1);
        switchTest(200);
        switchTest(-200);
        switchTest(70000);
        switchTest(-70000);
        switchTest(Long.MAX_VALUE);
    }

    private static void switchTest(long l) {
        switch (l) {
            case 0L -> System.out.println("zero");
            case byte n when n < 0 -> System.out.println("negative byte range");
            case byte n -> System.out.println("byte range");
            case short n when n < 0 -> System.out.println("negative short range");
            case short n -> System.out.println("short range");
            case int n when n < 0 -> System.out.println("negative int range");
            case int n -> System.out.println("int range");
            case long n -> System.out.println("long range");
        }
    }
}
