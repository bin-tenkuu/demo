package demo.ffm;

import demo.ffm.stdlib.Stdlib_h;
import demo.ffm.stdlib.lldiv_t;

import java.lang.foreign.Arena;

/**
 * @author bin
 * @since 2025/12/22
 */
public class StdlibTest {
    private static final Arena arena = Arena.ofAuto();

    static void main() {
        var lldiv = Stdlib_h.lldiv(arena, 1, 1);
        var quot = lldiv_t.quot(lldiv);
        var rem = lldiv_t.rem(lldiv);
        var start = System.currentTimeMillis();
        lldiv = Stdlib_h.lldiv(arena, 31558149L, 3600L);
        quot = lldiv_t.quot(lldiv);
        rem = lldiv_t.rem(lldiv);
        System.out.printf("Earth orbit: %d hours and %d seconds.\n", quot, rem);
        System.out.println(System.currentTimeMillis() - start);
    }
}
