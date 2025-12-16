package demo.constant;

/// 缓存的key 常量
public interface UserConstants {
    /// 登录用户 redis key
    String LOGIN_TOKEN_KEY = "login_tokens:";

    /// 令牌前缀
    String TOKEN_PREFIX = "Bearer ";

    /// 管理员用户ID
    long ADMIN_ID = 0L;
    /// 所有权限标识
    String ALL_PERMISSION = "*";

    /// 用户名长度限制
    int USERNAME_MIN_LENGTH = 2;
    int USERNAME_MAX_LENGTH = 20;

    /// 密码长度限制
    int PASSWORD_MIN_LENGTH = 8;
    int PASSWORD_MAX_LENGTH = 20;
}
