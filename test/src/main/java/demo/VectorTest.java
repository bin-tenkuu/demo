package demo;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * @author bin
 * @since 2026/08/18
 */
public class VectorTest {
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    static void main() {
        var a = new float[100];
        var b = new float[100];
        var c = new float[100];
        vectorComputation(a, b, c);
    }

    static void vectorComputation(float[] a, float[] b, float[] c) {
        for (int i = 0; i < a.length; i += SPECIES.length()) {
            var m = SPECIES.indexInRange(i, a.length);
            var va = FloatVector.fromArray(SPECIES, a, i, m);
            var vb = FloatVector.fromArray(SPECIES, b, i, m);
            var vc = va.mul(va)
                    .add(vb.mul(vb))
                    .neg();
            vc.intoArray(c, i, m);
        }
    }
}
