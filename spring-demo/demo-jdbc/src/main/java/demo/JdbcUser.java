package demo;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.springframework.data.relational.core.mapping.Table;

import java.time.LocalDateTime;

/**
 * @author bin
 * @since 2025/12/09
 */
@Getter
@Setter
@ToString
@Table
public class JdbcUser {
    private Long id;
    private String name;
    private LocalDateTime createdAt;
    private int age;
}
