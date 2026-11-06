package demo.core.util;

import lombok.Getter;
import org.jetbrains.annotations.Contract;

/**
 * @author bin
 * @since 2026/11/06
 */
public final class AverageTracker {
    private long count = 0;
    @Getter
    private double average = 0.0;

    /**
     * 输入耗时并返回当前平均值
     * 此方法不会发生数值溢出，适合极长期运行
     */
    @Contract(pure = true)
    public void addDuration(long durationMillis) {
        count++;
        // 核心算法：利用差值动态更新平均值
        average += (durationMillis - average) / count;
    }

}
