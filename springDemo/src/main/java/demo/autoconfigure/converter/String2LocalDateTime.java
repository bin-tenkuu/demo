package demo.autoconfigure.converter;

import demo.util.DateUtil;
import org.jetbrains.annotations.NotNull;
import org.springframework.core.convert.converter.Converter;

import java.time.LocalDateTime;

/**
 * String 转 LocalDateTime
 *
 * @author bin
 * @since 2022/12/28
 */
public class String2LocalDateTime implements Converter<String, LocalDateTime> {
    @Override
    public LocalDateTime convert(final @NotNull String source) {
        return DateUtil.parseLocalDateTime(source);
    }
}
