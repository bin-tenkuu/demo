package demo.贷款计算器;

import java.text.DecimalFormat;
import java.time.YearMonth;

/**
 * @author bin
 * @since 2026/01/16
 */
@SuppressWarnings("NonAsciiCharacters")
public class 提前还贷计算器 {
    private static final DecimalFormat df = new DecimalFormat();

    static {
        df.setMinimumFractionDigits(2);
        df.setMaximumFractionDigits(2);
        df.setGroupingUsed(false);
    }

    private static final double 贷款金额 = 30 * 10000;
    private static final int 贷款期数 = 10 * 12;
    private static final double 贷款年利率 = 2.6 / 100;
    private static final double 贷款月利率 = 贷款年利率 / 12;
    private static final 还款方式E 还款方式 = 还款方式E.等额本金;
    private static final YearMonth 首次还款时间 = YearMonth.of(2026, 1);
    private static final 提前还款[] 提前还款方案 = {
            new 提前还款(2026, 1, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 2, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 3, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 4, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 5, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 6, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 7, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 8, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 9, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 10, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 11, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2026, 12, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2027, 1, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2027, 2, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2027, 3, 10000, 提前还款方式E.减少月还款额),
            new 提前还款(2027, 4, 10000, 提前还款方式E.缩短还款期限),
            new 提前还款(2027, 5, 10000, 提前还款方式E.缩短还款期限),
    };

    private static class 报告 {
        public YearMonth 还款日期;
        public int 剩余期数;
        public double 还前本金;
        public double 还后本金;
        public double 偿还本金;
        public double 偿还利息;
        public double 还款金额;

        public 报告() {
            还款日期 = 首次还款时间;
            剩余期数 = 贷款期数;
            还前本金 = 贷款金额;
            偿还本金 = 还前本金 / 剩余期数;
            偿还利息 = 还前本金 * 贷款月利率;
            还后本金 = 还前本金 - 偿还本金;
            还款金额 = 偿还本金 + 偿还利息;
        }

        public 报告 next() {
            var nbg = new 报告();
            nbg.还款日期 = 还款日期.plusMonths(1);
            nbg.剩余期数 = 剩余期数 - 1;
            nbg.还前本金 = 还后本金;
            if (还款方式 == 还款方式E.等额本金) {
                nbg.偿还本金 = nbg.还前本金 / nbg.剩余期数;
                nbg.偿还利息 = nbg.还前本金 * 贷款月利率;
                nbg.还后本金 = nbg.还前本金 - nbg.偿还本金;
                nbg.还款金额 = nbg.偿还本金 + nbg.偿还利息;
            }
            return nbg;
        }

        public void 提前还款(提前还款方式E 方式, double 提前还款金额) {
            还前本金 = 还后本金;
            还后本金 = 还前本金 - 提前还款金额;
            if (还后本金 < 0.01) {
                // 提前结清
                剩余期数 = 0;
                偿还本金 = 0.0;
                偿还利息 = 0.0;
                还款金额 = 0.0;
                System.out.printf("--->\t%s\t\t%9s\t\t已结清\n", 还款日期, f(提前还款金额));
                return;
            }
            switch (方式) {
                case 减少月还款额 -> {
                    System.out.printf("--->\t%s\t\t%9s\t\t(减少月还款额)\t\t\t\t\t%9s\t->\t%9s\n",
                            还款日期, f(提前还款金额), f(还前本金), f(还后本金)
                    );
                }
                case 缩短还款期限 -> {
                    if (还款方式 == 还款方式E.等额本金) {
                        // 计算新的剩余期数
                        var 新剩余期数 = (int) Math.ceil(还后本金 / (还款金额 - 还后本金 * 贷款月利率));
                        // 更新剩余期数
                        var 原剩余期数 = 剩余期数;
                        // 确保新的期数至少为1
                        剩余期数 = Math.max(1, 新剩余期数);
                        System.out.printf(
                                "--->\t%s\t\t%9s\t(缩短还款期限) 剩余期数 %d -> %d 期\t%9s\t->\t%9s\n",
                                还款日期, f(提前还款金额), 原剩余期数, 剩余期数, f(还前本金), f(还后本金));
                    }
                }
            }
        }

        public void print(int 期次) {
            System.out.printf("%4s\t%s\t\t%9s\t\t%9s\t\t%9s\t\t%9s\t->\t%9s\n",
                    期次, 还款日期, f(还款金额), f(偿还本金), f(偿还利息), f(还前本金), f(还后本金)
            );
        }
    }

    static void main() {
        var index = 0;
        var 报告 = new 报告();
        var 提前还款Index = 0;
        var 总利息 = 0.0;
        printHead();
        index++;
        报告.print(index);
        总利息 += 报告.偿还利息;
        while (报告.剩余期数 > 1) {
            while (提前还款Index < 提前还款方案.length &&
                    !报告.还款日期.isBefore(提前还款方案[提前还款Index].提前还款时间)) {
                var 提前还款 = 提前还款方案[提前还款Index];
                报告.提前还款(提前还款.提前还款方案, 提前还款.提前还款金额);
                提前还款Index++;
            }
            报告 = 报告.next();
            index++;
            报告.print(index);
            总利息 += 报告.偿还利息;
        }
        System.out.printf("总利息: %.2f 元\n", 总利息);
    }

    public static void printHead() {
        System.out.printf("%s\t%s\t\t%6s\t\t%6s\t\t%6s\t\t%6s\t->\t%6s\n",
                "期次", "还款日期", "还款金额", "偿还本金", "偿还利息", "还前本金", "剩余本金");
    }

    private static String f(double v) {
        return df.format(v);
    }

    private enum 还款方式E {
        等额本息,
        等额本金
    }

    private enum 提前还款方式E {
        缩短还款期限,
        减少月还款额,
    }

    private record 提前还款(YearMonth 提前还款时间, double 提前还款金额, 提前还款方式E 提前还款方案) {
        public 提前还款(int year, int month, int 元, 提前还款方式E 提前还款方案) {
            this(YearMonth.of(year, month), 元, 提前还款方案);
        }
    }

}

