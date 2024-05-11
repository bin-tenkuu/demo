package demo;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;


public class SimdTest {
    public static void main() {
        final int size = 1024;
        final byte[] a = new byte[size];

        for (int i = 0; i < 1024; i += 2) {
            a[i] = (byte) 1;
            a[i + 1] = (byte) -1;
        }

        vectorComputation(a);

    }

    private static void vectorComputation(byte[] a) {
        final VectorSpecies<Byte> SPECIES = ByteVector.SPECIES_MAX;

        ByteVector vector = ByteVector.zero(SPECIES);
        for (int i = 0; i < a.length; i += SPECIES.length()) {
            var m = SPECIES.indexInRange(i, a.length);
            var va = ByteVector.fromArray(SPECIES, a, i, m)
                    .lanewise(VectorOperators.ASHR, 7)
                    .not()
                    .and(ByteVector.fromArray(SPECIES, a, i, m));
            vector = vector.add(va);
        }

        byte[] bs = new byte[SPECIES.length()];
        vector.intoArray(bs, 0);

        long sum = 0;
        for (byte b : bs) {
            sum += b;
        }
        System.out.println(sum);
    }
}
