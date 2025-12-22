package demo.ffm;

import demo.ffm.unistd.Unistd_h;

/**
 * @author bin
 * @since 2025/12/22
 */
public class UnistdTest {
    static void main() {
        var getuid = Unistd_h.getuid();
        System.out.println(getuid);
        var setuid = Unistd_h.setuid(0);
        System.out.println(setuid);
    }
}
