package demo;

import lombok.val;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.time.LocalDate;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/09
 */
public class MoneyCalc {
    /**
     * 日利率 0.01%
     */
    private static final BigDecimal 日利率 = new BigDecimal("1.0001");
    /**
     * 月增长 3000
     */
    private static final int 月增长 = 5000;
    /**
     * 本金 10000
     */
    private static final int 本金 = 100000;

    private static final DecimalFormat formatter = new DecimalFormat("#.#");

    static {
        formatter.setRoundingMode(RoundingMode.DOWN);
        formatter.setMinimumFractionDigits(0);
        formatter.setMaximumFractionDigits(16);
    }

    public static void main(String[] args) {

        BigDecimal v0 = calc(本金, new BigDecimal("1.0001"), 5000, 0);
        System.out.println(v0);
        BigDecimal last = v0, now;
        for (int year = 1; year <= 10; year++) {
            now = calc(本金, new BigDecimal("1.0002"), 5000, year);
            val subtract = now.subtract(last);
            System.out.printf("第 %s 年总：\t%s\t多:\t%s\t平均：\t%s\n", year, now, subtract,
                    now.subtract(v0).divide(new BigDecimal(year), 8, RoundingMode.DOWN));
            last = now;
        }
    }

    @SuppressWarnings("NonAsciiCharacters")
    private static BigDecimal calc(int 本金, BigDecimal 日利率, int 月增长, int year) {
        BigDecimal total = new BigDecimal(本金);
        LocalDate date = LocalDate.of(0, 1, 1);
        printf(date, total);
        boolean first = true;
        for (; date.getYear() < year; date = date.plusDays(1)) {
            total = total.multiply(日利率, MathContext.DECIMAL128).setScale(8, RoundingMode.DOWN);
            if (date.getDayOfMonth() == 1) {
                if (first) {
                    first = false;
                } else {
                    total = total.add(new BigDecimal(月增长)).setScale(8, RoundingMode.DOWN);
                }
            }
            printf(date, total);
        }
        return total;
    }

    private static void printf(LocalDate month, BigDecimal total) {
        // System.out.printf("%s：\t%s\n",
        //         month, formatter.format(total));
    }
}
