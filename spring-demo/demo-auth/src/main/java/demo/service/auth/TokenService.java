package demo.service.auth;

import demo.constant.UserConstants;
import demo.entity.SysUser;
import demo.entity.SysUserAuth;
import demo.exception.ResultException;
import demo.model.auth.LoginUser;
import demo.repository.SysMenuRepository;
import demo.repository.SysUserAuthRepository;
import demo.repository.SysUserRepository;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.springframework.http.HttpHeaders;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

/// @author bin
/// @since 2025/07/15
@Slf4j
@Component
@RequiredArgsConstructor
public class TokenService {
    private final Duration expireTimeDelay = Duration.ofMinutes(30);

    private final LocalCache localCache;
    private final SysUserRepository sysUserRepository;
    private final SysUserAuthRepository sysUserAuthRepository;
    private final SysMenuRepository sysMenuRepository;
    private final PasswordEncoder passwordEncoder;

    public String addExtryAuth(SysUserAuth auth) {
        val password = auth.getPassword();
        auth.setPassword(passwordEncoder.encode(password));
        sysUserAuthRepository.addExtryAuth(auth, expireTimeDelay);
        return password;
    }

    /// 登录前置校验
    public void loginPreCheck(String username, String password) {
        // 用户名或密码为空 错误
        if (!StringUtils.hasLength(username) || !StringUtils.hasLength(password)) {
            throw new ResultException("用户名或密码为空");
        }
        // 密码如果不在指定范围内 错误
        if (password.length() < UserConstants.PASSWORD_MIN_LENGTH
                || password.length() > UserConstants.PASSWORD_MAX_LENGTH) {
            throw new ResultException("密码不在指定范围内");
        }
        // 用户名不在指定范围内 错误
        if (username.length() < UserConstants.USERNAME_MIN_LENGTH
                || username.length() > UserConstants.USERNAME_MAX_LENGTH) {
            throw new ResultException("密码不在指定范围内");
        }
    }

    /// 登录验证
    ///
    /// @return token
    public String login(SysUserAuth auth) {
        // 登录前置校验
        var username = auth.getUsername();
        var password = auth.getPassword();
        loginPreCheck(username, password);
        var userAuth = switch (auth.getType()) {
            case STATIC -> sysUserAuthRepository.getExtryAuth(username);
            case DYNAMIC -> sysUserAuthRepository.findByUsername(username);
            case null -> null;
        };
        if (userAuth == null) {
            throw new UsernameNotFoundException("User not found: " + username);
        }
        var sysUser = sysUserRepository.getById(userAuth.getUserId());
        var user = LoginUser.from(userAuth.getUsername(), userAuth.getPassword(), sysUser);
        if (!user.isEnabled()) {
            log.info("登录用户：{} 已被停用.", username);
            throw new ResultException("对不起，您的账号：" + username + " 已停用");
        }
        // 用户验证
        if (!matches(password, user.getPassword())) {
            throw new ResultException("用户名或密码错误");
        }
        recordLoginInfo(user.getUser().getId());
        user.buildAuthorities(getMenuPermission(user.getUser()));
        // 生成token
        return createToken(user);
    }

    private void recordLoginInfo(Long userId) {
        SysUser sysUser = new SysUser();
        sysUser.setId(userId);
        // sysUser.setLoginIp(IpUtils.getIpAddr());
        sysUser.setLoginDate(LocalDateTime.now());
        // UserAgent userAgent = UserAgent.parseUserAgentString(ServletUtils.getRequest().getHeader("User-Agent"));
        // loginUser.setBrowser(userAgent.getBrowser().getName());
        // loginUser.setOs(userAgent.getOperatingSystem().getName());
        sysUserRepository.updateById(sysUser);
    }


    /// 获取用户身份信息
    public LoginUser getLoginUser(HttpServletRequest request) {
        // 获取请求携带的令牌
        String token = getToken(request);
        if (StringUtils.hasLength(token)) {
            try {
                String userKey = getTokenKey(token);
                return localCache.getCacheObject(userKey);
            } catch (Exception ignored) {
            }
        }
        return null;
    }

    /// 设置用户身份信息
    public void setLoginUser(LoginUser loginUser) {
        if (loginUser != null && StringUtils.hasLength(loginUser.getToken())) {
            refreshToken(loginUser);
        }
    }

    /// 删除用户身份信息
    public void delLoginUser(String token) {
        if (StringUtils.hasLength(token)) {
            String userKey = getTokenKey(token);
            localCache.deleteObject(userKey);
        }
    }

    /// 创建令牌
    public String createToken(LoginUser loginUser) {
        String token = UUID.randomUUID().toString().replace("-", "");
        loginUser.setToken(token);
        refreshToken(loginUser);
        return token;
    }

    /// 验证令牌有效期，相差不足20分钟，自动刷新缓存
    public void verifyToken(LoginUser loginUser) {
        long leftexpireTime = loginUser.getExpireTime() - System.currentTimeMillis();
        if (leftexpireTime <= Duration.ofMinutes(20).toMillis()) {
            refreshToken(loginUser);
        }
    }

    /// 刷新令牌有效期
    public void refreshToken(LoginUser loginUser) {
        var expireTime = expireTimeDelay.toMillis();
        loginUser.setExpireTime(System.currentTimeMillis() + expireTime);
        // 根据 uuid 将 loginUser 缓存
        String userKey = getTokenKey(loginUser.getToken());
        localCache.setCacheObject(userKey, loginUser, expireTime);
    }

    /// 获取请求 token
    private String getToken(HttpServletRequest request) {
        String token = request.getHeader(HttpHeaders.AUTHORIZATION);
        if (StringUtils.hasLength(token) && token.startsWith(UserConstants.TOKEN_PREFIX)) {
            token = token.replace(UserConstants.TOKEN_PREFIX, "");
        }
        return token;
    }

    private String getTokenKey(String uuid) {
        return UserConstants.LOGIN_TOKEN_KEY + uuid;
    }

    public String encode(String password) {
        return passwordEncoder.encode(password);
    }

    public boolean matches(String rawPassword, String encodedPassword) {
        return passwordEncoder.matches(rawPassword, encodedPassword);
    }

    /**
     * 获取菜单数据权限
     *
     * @return 菜单权限信息
     */
    public Set<String> getMenuPermission(SysUser user) {
        var menus = sysMenuRepository.listMenuByUserId(user.getId());
        Set<String> perms = new HashSet<>();
        for (var menu : menus) {
            perms.add(menu.getPerms());
        }
        return perms;
    }

}
