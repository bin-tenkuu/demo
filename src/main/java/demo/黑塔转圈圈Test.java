package demo;

import lombok.val;

import java.util.Arrays;
import java.util.function.BiConsumer;

/**
 * 黑塔转圈圈问题 TODO
 */
@SuppressWarnings({"NonAsciiCharacters", "preview"})
public class 黑塔转圈圈Test {
    // 题目
    // 黑塔女士的普通攻击可以对一名敌人造成一点伤害。此外，黑塔女士的天赋，会在敌人的生命值首次降低到初始值的二分之一（向下取整）时，立即发动追加攻击，对所有敌人造成一点伤害。
    //
    // 追加攻击造成的伤害可以继续触发追加攻击，且多个敌人的生命值同时下降到二分之一时，每个敌人都可以触发一次追加攻击。
    //
    // 现在有许多敌人，黑塔女士应该如何用最少的普通攻击消灭所有敌人呢？
    //
    // 数据格式
    // 输入
    // 一个长度为n整数数组a， 表示有n名敌人，第i名敌人初始生命值为a[i]。
    // 0 < 𝑛 < 10^5 ， 0 < 𝑎𝑖 < 10^9
    //
    // 输出
    // 一个整数，表示消灭所有敌人最少需要的普通攻击次数。
    // 示例
    //
    // [5, 6]
    // 7    // 需要普通攻击0号敌人3次，1号敌人4次。
    // [1, 2, 3, 4, 5]
    // 1    // 对0号敌人或者对1号敌人进行一次普通攻击即可。
    // [1, 3, 6]
    // 3    // 对初始生命为6的敌人进行3次普通攻击即可。
    //
    // 作者：xyzlancehe
    // 链接：https://leetcode.cn/circle/discuss/hrR0Gy/

    public static void main(String[] args) {
        BiConsumer<int[], Integer> test = (a, n) -> {
            val run = run(a);
            if (run != n) {
                System.out.println(STR."测试失败:\{n} != \{run} <- \{Arrays.toString(a)}");
            }
        };
        // 原题
        test.accept(new int[]{5, 6}, 7);
        test.accept(new int[]{1, 1, 1, 4}, 1);
        test.accept(new int[]{1, 1, 1, 1, 5}, 1);
        test.accept(new int[]{1, 2, 3, 4, 5}, 1);
        test.accept(new int[]{1, 3, 6}, 3);

        // 补充
        test.accept(new int[]{5, 5, 5, 5, 5}, 6);
    }

    private static int run(int[] a) {
        Arrays.sort(a);
        int n = a.length;
        int ret = 0;
        for (int an = 0; an < n; an++) {
            int i = a[an] - an - 1;
            int compare = i - (a[an] >> 1);
            if (compare > 1) {
                ret += i;
            } else if (compare > 0) {
                ret += i - 1;
            } else if (compare < -1) {
                // ret -= 1;
            }
        }
        return ret + 1;
    }

}
