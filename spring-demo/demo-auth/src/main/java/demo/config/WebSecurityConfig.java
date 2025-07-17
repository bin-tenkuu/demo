package demo.config;

import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.annotation.web.configurers.HeadersConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.DelegatingPasswordEncoder;
import org.springframework.security.crypto.password.NoOpPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.stereotype.Component;
import org.springframework.web.cors.CorsConfiguration;

import java.util.HashMap;
import java.util.Map;

/**
 * @author bin
 * @since 2025/07/15
 */
@SuppressWarnings("CodeBlock2Expr")
@Component
@RequiredArgsConstructor
public class WebSecurityConfig {
    private final HandlerAuthenticationFilter authenticationFilter;

    // 配置不同接口访问权限
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http.cors(configurer -> {
            configurer.configurationSource(request -> {
                CorsConfiguration config = new CorsConfiguration();
                config.setAllowCredentials(true);
                // 设置访问源地址
                config.addAllowedOriginPattern("*");
                // 设置访问源请求头
                config.addAllowedHeader("*");
                // 设置访问源请求方法
                config.addAllowedMethod("*");
                return config;
            });
        });
        http.csrf(AbstractHttpConfigurer::disable);
        http.headers(configurer -> {
            configurer.frameOptions(HeadersConfigurer.FrameOptionsConfig::sameOrigin);
        });
        // 使用自己实现的登陆和登出逻辑
        http.formLogin(AbstractHttpConfigurer::disable);
        http.logout(AbstractHttpConfigurer::disable);
        http.rememberMe(AbstractHttpConfigurer::disable);
        http.httpBasic(AbstractHttpConfigurer::disable);
        http.addFilterAt(authenticationFilter, UsernamePasswordAuthenticationFilter.class);
        http.exceptionHandling(config -> {
            // config.disable();
            config.authenticationEntryPoint(authenticationFilter);
            config.accessDeniedHandler(authenticationFilter);
        });
        // http.userDetailsService(sysUserRepository);
        http.sessionManagement(config -> {
            config.sessionCreationPolicy(SessionCreationPolicy.STATELESS);
        });
        http.authorizeHttpRequests((authorize) -> {
            authorize.requestMatchers(new String[]{
                    // "/",
                    "/**.html",
                    "/**.js",
                    "/**.css",
                    "/webjars/**",
                    "/v3/api-docs/**",
                    "/login",
                    "/register",
                    "/tempLoginApply",
            }).permitAll();
            // 除上面外的所有请求全部需要鉴权认证
            authorize.anyRequest()
                    // .permitAll();
                    .authenticated();
        });
        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        String encodingId = "bcrypt";
        Map<String, PasswordEncoder> encoders = new HashMap<>();
        encoders.put(encodingId, new BCryptPasswordEncoder());
        // noinspection deprecation
        encoders.put("noop", NoOpPasswordEncoder.getInstance());
        return new DelegatingPasswordEncoder(encodingId, encoders);
    }
}


