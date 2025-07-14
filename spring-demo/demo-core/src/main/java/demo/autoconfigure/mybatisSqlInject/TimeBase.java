package demo.autoconfigure.mybatisSqlInject;

import java.time.LocalDateTime;

/**
 * @author bin
 * @since 2025/05/06
 */
public interface TimeBase {
    LocalDateTime getTime();

    void setTime(LocalDateTime time);

    String getId();

    void setId(String id);
}
