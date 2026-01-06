package demo.util;

import org.jetbrains.annotations.Nullable;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * @author bin
 * @since 2025/12/23
 */
public final class NumberUtil {
    // region valueOf

    @Nullable
    public static BigDecimal valueOf(@Nullable String v) {
        return v == null ? null : new BigDecimal(v);
    }

    public static BigDecimal valueOf(double v) {
        return BigDecimal.valueOf(v);
    }

    @Nullable
    public static BigDecimal valueOf(@Nullable Double v) {
        return v == null ? null : BigDecimal.valueOf(v);
    }

    public static BigDecimal valueOf(float v) {
        return BigDecimal.valueOf(v);
    }

    @Nullable
    public static BigDecimal valueOf(@Nullable Float v) {
        return v == null ? null : BigDecimal.valueOf(v);
    }

    public static BigDecimal valueOf(long v) {
        return BigDecimal.valueOf(v);
    }

    @Nullable
    public static BigDecimal valueOf(@Nullable Long v) {
        return v == null ? null : BigDecimal.valueOf(v);
    }

    public static BigDecimal valueOf(int v) {
        return BigDecimal.valueOf(v);
    }

    @Nullable
    public static BigDecimal valueOf(@Nullable Integer v) {
        return v == null ? null : BigDecimal.valueOf(v);
    }

    public static BigDecimal valueOf(short v) {
        return BigDecimal.valueOf(v);
    }

    @Nullable
    public static BigDecimal valueOf(@Nullable Short v) {
        return v == null ? null : BigDecimal.valueOf(v);
    }

    public static BigDecimal valueOf(byte v) {
        return BigDecimal.valueOf(v);
    }

    @Nullable
    public static BigDecimal valueOf(@Nullable Byte v) {
        return v == null ? null : BigDecimal.valueOf(v);
    }

    public static BigDecimal valueOf(boolean v) {
        return BigDecimal.valueOf(v ? 1 : 0);
    }

    @Nullable
    public static BigDecimal valueOf(@Nullable Boolean v) {
        return v == null ? null : BigDecimal.valueOf(v ? 1 : 0);
    }

    // endregion
    // region scale

    @Nullable
    public static BigDecimal scale(@Nullable BigDecimal v, int scale) {
        return v == null ? null : v.setScale(scale, RoundingMode.HALF_UP);
    }

    @Nullable
    public static BigDecimal scale(@Nullable BigDecimal v, int scale, RoundingMode mode) {
        return v == null ? null : v.setScale(scale, mode);
    }

    // endregion
    // region default

    public static BigDecimal ifNull(BigDecimal v) {
        return v == null ? BigDecimal.ZERO : v;
    }

    public static BigDecimal ifNull(BigDecimal v, BigDecimal defaultValue) {
        return v == null ? defaultValue : v;
    }

    public static BigDecimal ifNull(BigDecimal v, double defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, Double defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, float defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, Float defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, long defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, Long defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, int defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, Integer defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, short defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, Short defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, byte defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, Byte defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, boolean defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    public static BigDecimal ifNull(BigDecimal v, Boolean defaultValue) {
        return v == null ? valueOf(defaultValue) : v;
    }

    // endregion
    // region add

    @Nullable
    public static BigDecimal add(@Nullable BigDecimal v1, @Nullable BigDecimal v2) {
        if (v1 == null) {
            return v2;
        }
        if (v2 == null) {
            return v1;
        }
        return v1.add(v2);
    }

    @Nullable
    public static BigDecimal add(@Nullable BigDecimal v1, @Nullable BigDecimal... vs) {
        var sum = v1;
        for (var v : vs) {
            if (v == null) {
                continue;
            }
            if (sum == null) {
                sum = v;
            } else {
                sum = sum.add(v);
            }
        }
        return sum;
    }

    // endregion
    // region sub
    @Nullable
    public static BigDecimal neg(@Nullable BigDecimal v) {
        return v == null ? null : v.negate();
    }

    @Nullable
    public static BigDecimal sub(@Nullable BigDecimal v1, @Nullable BigDecimal v2) {
        if (v2 == null) {
            return v1;
        }
        if (v1 == null) {
            return v2.negate();
        }
        return v1.subtract(v2);
    }

    // endregion
    // region mul

    @Nullable
    public static BigDecimal mul(@Nullable BigDecimal v1, @Nullable BigDecimal v2) {
        if (v1 == null || v2 == null) {
            return null;
        }
        return v1.multiply(v2);
    }

    public static BigDecimal mul(BigDecimal v1, BigDecimal... vs) {
        if (v1 == null) {
            return null;
        }
        var mul = v1;
        for (var v : vs) {
            if (v == null) {
                return null;
            }
            mul = mul.multiply(v);
        }
        return mul;
    }

    // endregion
    // region div

    @Nullable
    public static BigDecimal div(@Nullable BigDecimal v1, @Nullable BigDecimal v2) {
        if (v1 == null || v2 == null) {
            return null;
        }
        if (v2.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return v1.divide(v2, RoundingMode.HALF_UP);
    }

    @Nullable
    public static BigDecimal div(@Nullable BigDecimal v1, @Nullable BigDecimal v2, int scale) {
        if (v1 == null || v2 == null) {
            return null;
        }
        if (v2.compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return v1.divide(v2, scale, RoundingMode.HALF_UP);
    }

    // endregion
    // region other
    @Nullable
    public static BigDecimal max(@Nullable BigDecimal v1, @Nullable BigDecimal v2) {
        if (v1 == null) {
            return v2;
        }
        if (v2 == null) {
            return v1;
        }
        return v1.compareTo(v2) >= 0 ? v1 : v2;
    }

    @Nullable
    public static BigDecimal min(@Nullable BigDecimal v1, @Nullable BigDecimal v2) {
        if (v1 == null) {
            return v2;
        }
        if (v2 == null) {
            return v1;
        }
        return v1.compareTo(v2) <= 0 ? v1 : v2;
    }

    // endregion
}
