package demo.enums;

import lombok.val;
import org.jetbrains.annotations.NotNull;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.temporal.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/05
 */
@SuppressWarnings("unused")
public enum TimeType {
    year("Y", "yyyy'年'", ChronoUnit.YEARS, ChronoField.YEAR) {
        @Override
        public LocalDateTime getPart(LocalDateTime time) {
            return LocalDateTime.MIN.withYear(0);
        }

        @Override
        public LocalDateTime truncatedTo(LocalDateTime time) {
            return LocalDateTime.MIN.withYear(time.getYear());
        }
    },
    month("M", "MM'月'", ChronoUnit.MONTHS, ChronoField.MONTH_OF_YEAR) {
        @Override
        public LocalDateTime getPart(LocalDateTime time) {
            return LocalDateTime.MIN.withYear(time.getYear());
        }

        @Override
        public LocalDateTime truncatedTo(LocalDateTime time) {
            return time.toLocalDate().withDayOfMonth(1).atStartOfDay();
        }
    },
    day("D", "dd'日'", ChronoUnit.DAYS, ChronoField.DAY_OF_MONTH) {
        @Override
        public LocalDateTime getPart(LocalDateTime time) {
            return time.toLocalDate().withDayOfMonth(1).atStartOfDay();
        }

        @Override
        public LocalDateTime truncatedTo(LocalDateTime time) {
            return time.with(LocalTime.MIN);
        }
    },
    hour("HH", "HH'时'", ChronoUnit.HOURS, ChronoField.HOUR_OF_DAY) {
        @Override
        public LocalDateTime getPart(LocalDateTime time) {
            return time.with(LocalTime.MIN);
        }

        @Override
        public LocalDateTime truncatedTo(LocalDateTime time) {
            return time.truncatedTo(unit);
        }
    },
    min15("MI", "mm'分'", Minutes15.INSTANCE, Minute15OfHour.INSTANCE) {
        @Override
        public LocalDateTime getPart(LocalDateTime time) {
            return time.truncatedTo(ChronoUnit.HOURS);
        }

        @Override
        public LocalDateTime truncatedTo(LocalDateTime time) {
            return time.truncatedTo(unit);
        }
    },
    ;
    private static final TimeType[] VALUES = values();
    public final String type;
    public final DateTimeFormatter formater;
    public final TemporalUnit unit;
    public final TemporalField field;

    TimeType(String type, String formater, TemporalUnit unit, TemporalField field) {
        this.type = type;
        this.formater = DateTimeFormatter.ofPattern(formater);
        this.unit = unit;
        this.field = field;
    }

    /**
     * 舍弃低于或等于当前精度的时间
     */
    public abstract LocalDateTime getPart(LocalDateTime time);

    /**
     * 舍弃低于当前精度的时间
     */
    public abstract LocalDateTime truncatedTo(LocalDateTime time);

    public LocalDateTime plus(LocalDateTime time, long amountToAdd) {
        return time.plus(amountToAdd, unit);
    }

    public LocalDateTime plus(LocalDateTime time) {
        return time.plus(1, unit);
    }

    public TimeType getParent() {
        return switch (this) {
            case day -> month;
            case hour -> day;
            case min15 -> hour;
            default -> year;
        };
    }

    public static DateTimeFormatter getFormatter(TimeType from, TimeType to) {
        val builder = new DateTimeFormatterBuilder();
        for (int i = from.ordinal(); i <= to.ordinal(); i++) {
            builder.append(VALUES[i].formater);
        }
        return builder.toFormatter();
    }

    public List<String> toString(List<LocalDateTime> timeList) {
        val list = new ArrayList<String>(timeList.size());
        if (timeList.isEmpty()) {
            return list;
        }
        val startDate = timeList.getFirst();
        val endDate = timeList.getLast();
        TimeType from = getTimeTypeFrom(startDate, endDate);
        val pattern = getFormatter(from, this);
        for (LocalDateTime x : timeList) {
            list.add(pattern.format(x));
        }
        return list;
    }

    private @NotNull TimeType getTimeTypeFrom(LocalDateTime startDate, LocalDateTime endDate) {
        TimeType from;
        if (startDate.getYear() != endDate.getYear() || this == TimeType.year) {
            from = TimeType.year;
        } else if (startDate.getMonth() != endDate.getMonth() || this == TimeType.month) {
            from = TimeType.month;
        } else if (startDate.getDayOfMonth() != endDate.getDayOfMonth() || this == TimeType.day) {
            from = TimeType.day;
        } else if (startDate.getHour() != endDate.getHour() || this == TimeType.hour) {
            from = TimeType.hour;
        } else {
            from = TimeType.min15;
        }
        return from;
    }

    private static final class Minutes15 implements TemporalUnit {
        public static final Minutes15 INSTANCE = new Minutes15();

        @Override
        public Duration getDuration() {
            return Duration.ofMinutes(15);
        }

        @Override
        public boolean isDurationEstimated() {
            return false;
        }

        @Override
        public boolean isDateBased() {
            return false;
        }

        @Override
        public boolean isTimeBased() {
            return true;
        }

        @Override
        public boolean isSupportedBy(Temporal temporal) {
            return ChronoUnit.MINUTES.isSupportedBy(temporal);
        }

        @Override
        public <R extends Temporal> R addTo(R temporal, long amount) {
            return ChronoUnit.MINUTES.addTo(temporal, amount * 15);
        }

        @Override
        public long between(Temporal temporal1Inclusive, Temporal temporal2Exclusive) {
            val until = temporal1Inclusive.until(temporal2Exclusive, ChronoUnit.MINUTES);
            return until / 15 * 15;
        }
    }

    private static final class Minute15OfHour implements TemporalField {
        public static final Minute15OfHour INSTANCE = new Minute15OfHour();
        private final ValueRange range = ValueRange.of(0, 3);

        @Override
        public String getDisplayName(Locale locale) {
            return "Minute15OfHour";
        }

        @Override
        public TemporalUnit getBaseUnit() {
            return Minutes15.INSTANCE;
        }

        @Override
        public TemporalUnit getRangeUnit() {
            return ChronoUnit.HOURS;
        }

        @Override
        public ValueRange range() {
            return range;
        }

        @Override
        public boolean isDateBased() {
            return false;
        }

        @Override
        public boolean isTimeBased() {
            return true;
        }

        @Override
        public boolean isSupportedBy(TemporalAccessor temporal) {
            return ChronoField.MINUTE_OF_HOUR.isSupportedBy(temporal);
        }

        @Override
        public ValueRange rangeRefinedBy(TemporalAccessor temporal) {
            if (isSupportedBy(temporal)) {
                return range();
            }
            throw new UnsupportedTemporalTypeException("Unsupported field: Minute15OfHour");
        }

        @Override
        public long getFrom(TemporalAccessor temporal) {
            return ChronoField.MINUTE_OF_HOUR.getFrom(temporal) / 15;
        }

        @Override
        public <R extends Temporal> R adjustInto(R temporal, long newValue) {
            return ChronoField.MINUTE_OF_HOUR.adjustInto(temporal, newValue * 15);
        }

    }
}
