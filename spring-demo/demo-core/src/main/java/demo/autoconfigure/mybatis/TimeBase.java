package demo.autoconfigure.mybatis;

import java.time.LocalDateTime;

/// @author bin
/// @since 2025/05/06
public interface TimeBase {
    String TIME = "TIME";
    String ID = "ID";

    LocalDateTime getTime();

    void setTime(LocalDateTime time);

    String getId();

    void setId(String id);
}
