
package demo.util;

import demo.enums.TimeType;
import lombok.val;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.time.temporal.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.function.Function;

/**
 * 时间工具类
 *
 * @author bin
 */
@SuppressWarnings("unused")
public class DateUtil {
    @NotNull
    @Contract("_ -> new")
    public static LocalDateTime toLocalDateTime(@NotNull Date date) {
        return toLocalDateTime(date.toInstant());
    }

    @NotNull
    @Contract("_ -> new")
    public static LocalDateTime toLocalDateTime(long time) {
        return toLocalDateTime(Instant.ofEpochMilli(time));
    }

    @NotNull
    @Contract("_ -> new")
    public static LocalDateTime toLocalDateTime(@NotNull Instant instant) {
        return LocalDateTime.ofInstant(instant, ZoneId.systemDefault());
    }

    @NotNull
    @Contract("_ -> new")
    public static Date toDate(@NotNull LocalDateTime date) {
        final ZoneId zoneId = ZoneId.systemDefault();
        final Instant instant = date.atZone(zoneId).toInstant();
        return Date.from(instant);
    }

    @NotNull
    public static Long toMilli(@NotNull LocalDateTime date) {
        final ZoneId zoneId = ZoneId.systemDefault();
        final Instant instant = date.atZone(zoneId).toInstant();
        return instant.toEpochMilli();
    }

    /**
     * 日期型字符串转化为日期 格式
     *
     * @param str 日期字符串：`2021`, `2021-02`, `2021-02-03`, `2021-02-03 04`, `2021-02-03 04:05`, `2021-02-03 04:05:06`
     * @return Date
     */
    @Nullable
    @Contract(value = "!null -> !null;", pure = true)
    public static Date parseDate(String str) {
        final LocalDateTime date = parseLocalDateTime(str);
        if (date == null) {
            return null;
        }
        return toDate(date);
    }

    private static final DateTimeFormatter DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern(
            "yyyy[-MM[-dd[ HH[:mm[:ss]]]]]"
    );

    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH[:mm[:ss]]");

    /**
     * 日期型字符串转化为日期 格式
     *
     * @param source 日期字符串：`2021`, `2021-02`, `2021-02-03`, `2021-02-03 04`, `2021-02-03 04:05`, `2021-02-03 04:05:06`
     * @return LocalDateTime
     */
    @Nullable
    @Contract(value = "!null -> !null;", pure = true)
    public static LocalDateTime parseLocalDateTime(String source) {
        if (source == null) {
            return null;
        }
        if (source.isEmpty()) {
            return LocalDateTime.now();
        }
        try {
            return DATE_TIME_FORMATTER.parse(source, SAFE_LOCAL_DATE_TIME);
        } catch (DateTimeParseException e) {
            throw new IllegalArgumentException("不受支持的时间格式：" + source);
        }
    }

    /**
     * 日期型字符串转化为日期 格式
     *
     * @param source 日期字符串：`2021`, `2021-02`, `2021-02-03`
     * @return LocalDate
     */
    @Nullable
    @Contract(value = "!null -> !null;", pure = true)
    public static LocalDate parseLocalDate(String source) {
        if (source == null) {
            return null;
        }
        if (source.isEmpty()) {
            return LocalDate.now();
        }
        try {
            return DATE_TIME_FORMATTER.parse(source, SAFE_LOCAL_DATE);
        } catch (DateTimeParseException e) {
            throw new IllegalArgumentException("不受支持的时间格式：" + source);
        }
    }

    /**
     * 日期型字符串转化为日期 格式
     *
     * @param source 日期字符串：`04`, `04:05`, `04:05:06`
     * @return LocalTime
     */
    @Nullable
    @Contract(value = "!null -> !null;", pure = true)
    public static LocalTime parseLocalTime(String source) {
        if (source == null) {
            return null;
        }
        if (source.isEmpty()) {
            return LocalTime.now();
        }
        try {
            return TIME_FORMATTER.parse(source, SAFE_LOCAL_TIME);
        } catch (DateTimeParseException e) {
            throw new IllegalArgumentException("不受支持的时间格式：" + source);
        }
    }

    public static void truncatedTo(Instant instant, TemporalUnit unit) {
        instant.truncatedTo(unit);
    }

    private static final TemporalQuery<LocalDate> LOCAL_DATE = temporal -> {
        final int[] ints = new int[]{0, 1};
        if (temporal.isSupported(ChronoField.YEAR)) {
            ints[0] = temporal.get(ChronoField.YEAR);
            if (temporal.isSupported(ChronoField.MONTH_OF_YEAR)) {
                ints[1] = temporal.get(ChronoField.MONTH_OF_YEAR);
            }
        }
        return LocalDate.of(ints[0], ints[1], 1);
    };

    private static final TemporalQuery<LocalTime> LOCAL_TIME = temporal -> {
        final int[] ints = new int[3];
        if (temporal.isSupported(ChronoField.HOUR_OF_DAY)) {
            ints[0] = temporal.get(ChronoField.HOUR_OF_DAY);
            if (temporal.isSupported(ChronoField.MINUTE_OF_HOUR)) {
                ints[1] = temporal.get(ChronoField.MINUTE_OF_HOUR);
            }
        }
        return LocalTime.of(ints[0], ints[1], 0);
    };

    public static final TemporalQuery<LocalDate> SAFE_LOCAL_DATE = temporal -> {
        final LocalDate localDate = temporal.query(TemporalQueries.localDate());
        if (localDate == null) {
            return LOCAL_DATE.queryFrom(temporal);
        }
        return localDate;
    };

    public static final TemporalQuery<LocalTime> SAFE_LOCAL_TIME = temporal -> {
        final LocalTime localDate = temporal.query(TemporalQueries.localTime());
        if (localDate == null) {
            return LOCAL_TIME.queryFrom(temporal);
        }
        return localDate;
    };

    public static final TemporalQuery<LocalDateTime> SAFE_LOCAL_DATE_TIME = temporal -> {
        final LocalDate localDate = temporal.query(TemporalQueries.localDate());
        if (localDate == null) {
            return LocalDateTime.of(LOCAL_DATE.queryFrom(temporal), LocalTime.MIN);
        }
        final LocalTime localTime = temporal.query(TemporalQueries.localTime());
        if (localTime == null) {
            return LocalDateTime.of(localDate, LOCAL_TIME.queryFrom(temporal));
        }
        return LocalDateTime.of(localDate, localTime);
    };

    public static List<LocalDate> toDayList(LocalDate start, LocalDate end) {
        val list = new ArrayList<LocalDate>();
        for (LocalDate time = start; !time.isAfter(end); time = time.plusDays(1)) {
            list.add(time);
        }
        return list;
    }

    public static final Month[] MONTHS = Month.values();

    public static List<LocalDate> toMonthList(int year) {
        val list = new ArrayList<LocalDate>(MONTHS.length);
        for (Month value : MONTHS) {
            list.add(LocalDate.of(year, value, 1));
        }
        return list;
    }

    public static LocalDateTime format(TimeType type, LocalDateTime time) {
        switch (type) {
            case year -> {
                return LocalDate.of(time.getYear(), 1, 1).atStartOfDay();
            }
            case month -> {
                return time.toLocalDate().withDayOfMonth(1).atStartOfDay();
            }
            case day -> {
                return time.toLocalDate().atStartOfDay();
            }
            case hour -> {
                return time.truncatedTo(ChronoUnit.HOURS);
            }
            case min15 -> {
                val dateTime = time.truncatedTo(ChronoUnit.MINUTES);
                return dateTime.withMinute(dateTime.getMinute() / 15 * 15);
            }
            default -> throw new IllegalStateException("never");
        }
    }

    /**
     * 获取同层级时间列表
     */
    public static List<LocalDateTime> toList(TimeType type, LocalDateTime time) {
        val list = new ArrayList<LocalDateTime>(MONTHS.length);
        switch (type) {
            case month -> {
                val year = time.getYear();
                for (Month value : MONTHS) {
                    list.add(LocalDate.of(year, value, 1).atStartOfDay());
                }
            }
            case day -> {
                val date = time.toLocalDate().with(TemporalAdjusters.lastDayOfMonth());
                for (int i = 1; i < date.getDayOfMonth(); i++) {
                    list.add(date.withDayOfMonth(i).atStartOfDay());
                }
            }
            case hour -> {
                val date = time.toLocalDate().atStartOfDay();
                for (int i = 0; i < 24; i++) {
                    list.add(date.withHour(i));
                }
            }
        }
        return list;
    }

    public static List<LocalDateTime> toList(TimeType type, LocalDateTime start, LocalDateTime end) {
        val list = new ArrayList<LocalDateTime>(MONTHS.length);
        Function<LocalDateTime, LocalDateTime> nextTime = switch (type) {
            case year -> time -> time.plusYears(1);
            case month -> time -> time.plusMonths(1);
            case day -> time -> time.plusDays(1);
            case hour -> time -> time.plusHours(1);
            case min15 -> time -> time.plusMinutes(15);
        };
        LocalDateTime time = format(type, start);
        // if (!time.isBefore(start)) {
        //     list.add(time);
        //     time = nextTime.apply(time);
        // }
        while (!time.isAfter(end)) {
            list.add(time);
            time = nextTime.apply(time);
        }
        return list;
    }

}
