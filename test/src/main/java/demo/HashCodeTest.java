package demo;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/01/22
 */
public class HashCodeTest {
    /**
     * -XX:+UnlockExperimentalVMOptions -XX:hashCode=2
     */
    public static void main(String[] args) {
        System.out.println(new Object().hashCode());
        System.out.println(new Object().hashCode());
        System.out.println(new Object().hashCode());
        System.out.println(new Object().hashCode());
        System.out.println(new Object().hashCode());
        System.out.println(new Object().hashCode());
        System.out.println(new Object().hashCode());
        System.out.println("polygenelubricants".hashCode());
        System.out.println("GydZG_".hashCode());
        System.out.println("DESIGNING WORKHOUSES".hashCode());
    }
}
