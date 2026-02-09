package demo.service.auth;

import demo.model.ResultModel;
import demo.model.auth.LoginUser;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.security.autoconfigure.UserDetailsServiceAutoConfiguration;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.AuthenticationEntryPoint;
import org.springframework.security.web.access.AccessDeniedHandler;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import tools.jackson.databind.json.JsonMapper;

import java.io.IOException;

/**
 * @author bin
 * @since 2025/07/15
 */
@Component
@RequiredArgsConstructor
@EnableAutoConfiguration(exclude = UserDetailsServiceAutoConfiguration.class)
public class HandlerAuthenticationFilter extends OncePerRequestFilter
        implements AuthenticationEntryPoint, AccessDeniedHandler {
    private final TokenService tokenService;
    private final JsonMapper jsonMapper;

    @Override
    protected void doFilterInternal(
            @NotNull HttpServletRequest request,
            @NotNull HttpServletResponse response,
            @NotNull FilterChain chain
    ) throws ServletException, IOException {
        if (SecurityContextHolder.getContext().getAuthentication() == null) {
            LoginUser loginUser = tokenService.getLoginUser(request);
            if (loginUser != null) {
                tokenService.verifyToken(loginUser);
                UsernamePasswordAuthenticationToken authenticationToken = new UsernamePasswordAuthenticationToken(
                        loginUser, null, loginUser.getAuthorities()
                );
                authenticationToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                SecurityContextHolder.getContext().setAuthentication(authenticationToken);
            }
        }
        chain.doFilter(request, response);
    }

    /**
     * 未登陆
     */
    @Override
    public void commence(HttpServletRequest request, HttpServletResponse response,
            @NonNull AuthenticationException authException) throws IOException {
        String msg = String.format("请求访问：%s，认证失败", request.getRequestURI());
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType("application/json;charset=utf-8");
        response.getWriter().write(jsonMapper.writeValueAsString(ResultModel.fail(msg)));
    }

    /**
     * 未授权
     */
    @Override
    public void handle(HttpServletRequest request, HttpServletResponse response,
            @NonNull AccessDeniedException accessDeniedException) throws IOException {
        String msg = String.format("请求访问：%s，没有对应的权限", request.getRequestURI());
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType("application/json;charset=utf-8");
        response.getWriter().write(jsonMapper.writeValueAsString(ResultModel.fail(msg)));
    }

}
