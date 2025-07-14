package demo.autoconfigure.converter;

import demo.util.DateUtil;
import org.jetbrains.annotations.NotNull;
import org.springframework.core.convert.converter.Converter;

import java.time.LocalTime;

/**
 * String 转 LocalTime
 *
 * @author bin
 * @since 2022/12/28
 */
public class String2LocalTime implements Converter<String, LocalTime> {
    @Override
    public LocalTime convert(final @NotNull String source) {
        return DateUtil.parseLocalTime(source);
    }
}
