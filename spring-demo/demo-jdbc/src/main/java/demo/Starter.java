package demo;

import lombok.AllArgsConstructor;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * @author bin
 * @since 2025/12/09
 */
@AllArgsConstructor
@SpringBootApplication
public class Starter {
    private final JdbcUserRepository jdbcUserRepository;

    public static void main(String[] args) {
        SpringApplication.run(Starter.class, args);
    }
}
