package demo.auth.util;

import demo.auth.entity.SysUser;
import demo.auth.model.auth.LoginUser;
import demo.core.exception.ResultException;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.Optional;

/**
 * 安全服务工具类
 *
 * @author ruoyi
 */
public class SecurityUtils {
    /**
     * 用户ID
     **/
    public static Optional<Long> getUserId() {
        try {
            return getLoginUser()
                    .map(LoginUser::getUser)
                    .map(SysUser::getId);
        } catch (Exception e) {
            throw new ResultException("获取用户ID异常");
        }
    }

    /**
     * 获取用户账户
     **/
    public static Optional<String> getUsername() {
        return getLoginUser()
                .map(LoginUser::getUsername);
    }

    /**
     * 获取用户
     */
    public static Optional<LoginUser> getLoginUser() {
        try {
            var authentication = getAuthentication();
            if (authentication == null) {
                return Optional.empty();
            }
            var principal = authentication.getPrincipal();
            return Optional.ofNullable(principal)
                    .filter(k -> principal instanceof LoginUser)
                    .map(k -> (LoginUser) k);
        } catch (Exception e) {
            throw new ResultException("获取用户信息异常", e);
        }
    }

    /**
     * 获取Authentication
     */
    public static Authentication getAuthentication() {
        return SecurityContextHolder.getContext().getAuthentication();
    }

}
