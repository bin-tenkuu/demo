package demo.entity;

import demo.autoconfigure.mybatis.TimeBase;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * @author bin
 * @since 2025/05/06
 */
@Getter
@Setter
public class UserData implements TimeBase {
    private LocalDateTime time;
    private String id;
    private Double score;
}
