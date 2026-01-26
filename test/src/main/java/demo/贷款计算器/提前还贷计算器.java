package demo.贷款计算器;

import lombok.AllArgsConstructor;
import org.jetbrains.annotations.NotNull;

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
            new 提前还款(2026, 1, 2000, 提前还款方式E.减少月还款额),
    };

    @AllArgsConstructor
    private static class 报告 {
        public final YearMonth 还款日期;
        public final int 剩余期数;
        public final double 还前本金;
        public final double 还后本金;
        public final double 偿还本金;
        public final double 偿还利息;
        public final double 还款金额;

        public 报告(YearMonth 还款日期, int 剩余期数, double 还前本金) {
            this.还款日期 = 还款日期;
            this.剩余期数 = 剩余期数;
            this.还前本金 = 还前本金;
            this.偿还本金 = 还前本金 / 剩余期数;
            this.偿还利息 = 还前本金 * 贷款月利率;
            this.还后本金 = 还前本金 - 偿还本金;
            this.还款金额 = 偿还本金 + 偿还利息;
        }

        public 报告(YearMonth 还款日期, int 剩余期数, double 还前本金, double 偿还本金) {
            this.还款日期 = 还款日期;
            this.剩余期数 = 剩余期数;
            this.还前本金 = 还前本金;
            this.偿还本金 = 偿还本金;
            this.偿还利息 = 0;
            this.还后本金 = 还前本金 - 偿还本金;
            this.还款金额 = 偿还本金;
        }

        public 报告() {
            this(首次还款时间, 贷款期数, 贷款金额);
        }

        @NotNull
        public 报告 next() {
            if (剩余期数 <= 0 || 还后本金 < 0.01) {
                return new 报告(
                        还款日期.plusMonths(1),
                        0,
                        0
                );
            } else if (还款方式 == 还款方式E.等额本金) {
                return new 报告(
                        还款日期.plusMonths(1),
                        剩余期数 - 1,
                        还后本金
                );
            } else {
                return null;
            }
        }

        @NotNull
        public 报告 提前还款(Object 期次, 提前还款方式E 方式, double 提前还款金额) {
            报告 nbg;
            if (还后本金 <= 提前还款金额) {
                // 提前结清
                System.out.printf("%4s\t%s\t\t%9s\t\t已结清\n",
                        期次, 还款日期, f(提前还款金额)
                );
                return new 报告(
                        还款日期,
                        0,
                        还后本金,
                        还后本金
                );
            }
            switch (方式) {
                case 减少月还款额 -> {
                    if (还款方式 == 还款方式E.等额本金) {
                        nbg = new 报告(
                                还款日期,
                                剩余期数,
                                还后本金 - 提前还款金额
                        );
                        var 减少月还款额 = 偿还本金 - nbg.偿还本金;
                        // System.out.printf("%4s\t%s\t\t%9s\t\t(减少月还款额) 少 %s 元\t\t%9s\t->\t%9s\n",
                        //         期次, 还款日期, f(提前还款金额), f(减少月还款额), f(nbg.还前本金), f(nbg.还后本金)
                        // );
                        return nbg;
                    }
                }
                case 缩短还款期限 -> {
                    if (还款方式 == 还款方式E.等额本金) {
                        // 计算新的剩余期数
                        var 新剩余期数 = (int) Math.ceil(还后本金 / (还款金额 - 还后本金 * 贷款月利率));
                        // 更新剩余期数
                        // 确保新的期数至少为1
                        nbg = new 报告(
                                还款日期,
                                Math.max(2, 新剩余期数),
                                还后本金 - 提前还款金额
                        );
                        // System.out.printf(
                        //         "%4s\t%s\t\t%9s\t(缩短还款期限) 剩余期数 %d -> %d 期\t%9s\t->\t%9s\n",
                        //         期次, 还款日期, f(提前还款金额), 剩余期数, nbg.剩余期数, f(nbg.还前本金),
                        //         f(nbg.还后本金));
                        return nbg;
                    }
                }
            }
            return null;
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
        // var 提前还款Index = 0;
        var 总利息 = 0.0;
        var 剩余钱1 = 1000.0;
        var 剩余钱2 = 1000.0;
        printHead();
        while (报告.剩余期数 > 0) {
            index++;
            报告.print(index);
            总利息 += 报告.偿还利息;
            if (剩余钱2 > 0) {
                var next = 报告.提前还款(index, 提前还款方式E.缩短还款期限, 剩余钱2);
                // 剩余钱2 += 报告.偿还本金 - next.偿还本金;
                报告 = next;
            }
            if (剩余钱1 > 0) {
                var next = 报告.提前还款(index, 提前还款方式E.减少月还款额, 剩余钱1);
                // 剩余钱2 += 报告.偿还本金 - next.偿还本金;
                报告 = next;
            }
            // while (提前还款Index < 提前还款方案.length &&
            //         !报告.还款日期.isBefore(提前还款方案[提前还款Index].提前还款时间)) {
            //     var 提前还款 = 提前还款方案[提前还款Index];
            //     报告 = 报告.提前还款(提前还款.提前还款方案, 提前还款.提前还款金额);
            //     提前还款Index++;
            // }
            报告 = 报告.next();
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

    private static String f(int v) {
        return String.format("%4s", v);
    }

    private enum 还款方式E {
        等额本息,
        等额本金
    }

    private enum 提前还款方式E {
        缩短还款期限,
        减少月还款额,
    }

    private record 提前还款(
            YearMonth 提前还款时间,
            double 提前还款金额,
            提前还款方式E 提前还款方案
    ) {
        public 提前还款(int year, int month, double 元, 提前还款方式E 提前还款方案) {
            this(YearMonth.of(year, month), 元, 提前还款方案);
        }
    }

}

