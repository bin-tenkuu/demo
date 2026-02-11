package demo.core.autoconfigure.converter;

import demo.core.util.DateUtil;
import org.jetbrains.annotations.NotNull;
import org.springframework.core.convert.converter.Converter;
import org.springframework.stereotype.Component;

import java.time.LocalDate;

/// String 转 LocalDate
///
/// @author bin
/// @since 2022/12/28
@Component
public class String2LocalDate implements Converter<String, LocalDate> {
    @Override
    public LocalDate convert(final @NotNull String source) {
        return DateUtil.parseLocalDate(source);
    }
}
