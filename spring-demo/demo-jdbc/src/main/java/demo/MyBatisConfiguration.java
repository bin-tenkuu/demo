package demo;

import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.data.jdbc.repository.config.EnableJdbcRepositories;
import org.springframework.data.jdbc.repository.config.MyBatisJdbcConfiguration;

@Configuration
@EnableJdbcRepositories
@Import(MyBatisJdbcConfiguration.class)
public class MyBatisConfiguration {
}
