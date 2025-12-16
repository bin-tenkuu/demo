package demo.config;

import demo.service.auth.HandlerAuthenticationFilter;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.annotation.web.configurers.HeadersConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.stereotype.Component;
import org.springframework.web.cors.CorsConfiguration;

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
        // 禁用默认的表单登录
        http.formLogin(AbstractHttpConfigurer::disable);
        // 禁用默认的注销处理
        http.logout(AbstractHttpConfigurer::disable);
        // 禁用记住我功能
        http.rememberMe(AbstractHttpConfigurer::disable);
        // 禁用匿名用户
        http.anonymous(AbstractHttpConfigurer::disable);
        // 禁用HTTP Basic认证
        http.httpBasic(AbstractHttpConfigurer::disable);
        http.addFilterBefore(authenticationFilter, UsernamePasswordAuthenticationFilter.class);
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
                    "/",
                    "/**.html",
                    "/**.js",
                    "/**.css",
                    "/webjars/**",
                    "/v3/api-docs/**",
                    "/login",
                    "/register",
            }).permitAll();
            // 除上面外的所有请求全部需要鉴权认证
            authorize.anyRequest()
                    .permitAll();
                    // .authenticated();
        });
        return http.build();
    }

}


