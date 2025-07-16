package demo.service;

import demo.model.LoginUser;
import demo.util.CacheMap;
import jakarta.servlet.http.HttpServletRequest;
import lombok.val;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.time.Duration;
import java.util.UUID;

/**
 * @author bin
 * @since 2025/07/15
 */
@Component
public class TokenService {
    /**
     * 令牌前缀
     */
    private static final String TOKEN_PREFIX = "Bearer ";
    /**
     * 登录用户 redis key
     */
    private static final String LOGIN_TOKEN_KEY = "login_tokens:";

    private final CacheMap<String, LoginUser> cacheMap = new CacheMap<>();
    private final Duration expireTimeDelay = Duration.ofMinutes(30);

    /**
     * 获取用户身份信息
     *
     * @return 用户信息
     */
    public LoginUser getLoginUser(HttpServletRequest request) {
        // 获取请求携带的令牌
        String token = getToken(request);
        if (StringUtils.hasLength(token)) {
            try {
                String userKey = getTokenKey(token);
                return cacheMap.get(userKey);
            } catch (Exception ignored) {
            }
        }
        return null;
    }

    /**
     * 设置用户身份信息
     */
    public void setLoginUser(LoginUser loginUser) {
        if (loginUser != null && StringUtils.hasLength(loginUser.getToken())) {
            refreshToken(loginUser);
        }
    }

    /**
     * 删除用户身份信息
     */
    public void delLoginUser(String token) {
        if (StringUtils.hasLength(token)) {
            String userKey = getTokenKey(token);
            cacheMap.remove(userKey);
        }
    }

    /**
     * 创建令牌
     *
     * @param loginUser 用户信息
     * @return 令牌
     */
    public String createToken(LoginUser loginUser) {
        String token = UUID.randomUUID().toString().replace("-", "");
        loginUser.setToken(token);
        refreshToken(loginUser);
        return token;
    }

    /**
     * 验证令牌有效期，相差不足20分钟，自动刷新缓存
     */
    public void verifyToken(LoginUser loginUser) {
        long leftexpireTime = loginUser.getExpireTime() - System.currentTimeMillis();
        if (leftexpireTime <= Duration.ofMinutes(20).toMillis()) {
            refreshToken(loginUser);
        }
    }

    /**
     * 刷新令牌有效期
     *
     * @param loginUser 登录信息
     */
    public void refreshToken(LoginUser loginUser) {
        val expireTime = expireTimeDelay.toMillis();
        loginUser.setExpireTime(System.currentTimeMillis() + expireTime);
        // 根据uuid将loginUser缓存
        String userKey = getTokenKey(loginUser.getToken());
        cacheMap.set(userKey, loginUser, expireTime);
    }

    /**
     * 获取请求token
     *
     * @return token
     */
    private String getToken(HttpServletRequest request) {
        String token = request.getHeader(HttpHeaders.AUTHORIZATION);
        if (StringUtils.hasLength(token) && token.startsWith(TOKEN_PREFIX)) {
            token = token.replace(TOKEN_PREFIX, "");
        }
        return token;
    }

    private String getTokenKey(String uuid) {
        return LOGIN_TOKEN_KEY + uuid;
    }
}
