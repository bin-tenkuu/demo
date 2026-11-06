package demo.core.task;

import demo.core.enums.TimeType;
import demo.core.util.AverageTracker;
import demo.core.util.Threads;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentSkipListMap;

/**
 * @author bin
 * @since 2026/10/28
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class FormulaTask {
    private static final ConcurrentSkipListMap<LocalDateTime, Boolean> calcTimeDay = new ConcurrentSkipListMap<>();
    private static final AverageTracker averageTracker = new AverageTracker();
    private static volatile boolean running = false;
    private static FormulaTask instance;

    {
        instance = this;
    }

    public static void clear() {
        calcTimeDay.clear();
    }

    public static List<Object> getCalcTime() {
        var list = new ArrayList<>();
        if (calcTimeDay.size() < 5) {
            list.addAll(calcTimeDay.keySet());
        } else {
            var iterator = calcTimeDay.keySet().iterator();
            var curr = iterator.next();
            var sb = new StringBuilder();
            while (iterator.hasNext()) {
                var count = 0;
                sb.setLength(0);
                sb.append(curr);
                while (iterator.hasNext()) {
                    var next = iterator.next();
                    if (curr.until(next, ChronoUnit.MINUTES) == 15) {
                        curr = next;
                        count++;
                        continue;
                    }
                    curr = next;
                    break;
                }
                if (count != 0) {
                    sb.append(" ~ ").append(curr).append(", +15M * ").append(count);
                }
                list.add(sb.toString());
            }
        }
        var seconds = calcTimeDay.size() * averageTracker.getAverage() / 1000;
        list.add("预期计算完成时间：" + LocalDateTime.now().plusSeconds((long) seconds));
        return list;
    }

    public static void addDateTime(LocalDateTime time) {
        var curr = TimeType.min15.truncatedTo(time);
        calcTimeDay.put(curr, Boolean.TRUE);
    }

    public static void start() {
        if (running) {
            return;
        }
        synchronized (FormulaTask.class) {
            if (running) {
                return;
            }
            running = true;
        }
        Threads.execute(FormulaTask::run);
    }

    private static void run() {
        log.info("公式计算：启动");
        try {
            while (!calcTimeDay.isEmpty()) {
                var date = calcTimeDay.firstKey();
                calcTimeDay.remove(date);
                instance.calc(date);
            }
        } catch (InterruptedException e) {
            log.info("公式计算：线程中断", e);
        } catch (Exception e) {
            log.info("公式计算：未知错误", e);
        } finally {
            running = false;
            log.info("公式计算：结束");
        }
    }

    @Scheduled(cron = "0 5/15 * * * *")
    public void every15Minute() {
        addDateTime(LocalDateTime.now());
        start();
    }

    private void calc(LocalDateTime curr) throws InterruptedException {
        var start = System.currentTimeMillis();
        // TODO: 实际耗时操作
        Thread.sleep(1000);
        var duration = System.currentTimeMillis() - start;
        averageTracker.addDuration(duration);
        log.info("公式计算：计算耗时：{} ms, 平均耗时：{}ms", duration, averageTracker.getAverage());
    }

}
