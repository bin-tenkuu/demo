package demo.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * @author bin
 * @since 2025/07/25
 */
@Getter
@Setter
@Component
@ConfigurationProperties("auth-config")
public class AuthConfig {
    private String loginUrl;
    private List<String> permitAll;
}
