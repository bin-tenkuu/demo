package demo.model;

import demo.entity.SysUser;
import lombok.Getter;
import lombok.Setter;
import lombok.val;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author bin
 * @since 2025/07/15
 */
@Getter
@Setter
public class LoginUser implements UserDetails {
    private String username;
    private String password;
    private List<? extends GrantedAuthority> authorities;
    private SysUser user;

    private String token;
    private long loginTime;
    private long expireTime;

    public LoginUser() {
    }

    public static LoginUser from(String username, String password, SysUser user, List<String> authorities) {
        val list = authorities.stream().map(SimpleGrantedAuthority::new).collect(Collectors.toList());
        val loginUser = new LoginUser();
        loginUser.setUsername(username);
        loginUser.setPassword(password);
        loginUser.setUser(user);
        loginUser.setAuthorities(list);
        loginUser.setLoginTime(System.currentTimeMillis());
        return loginUser;
    }

    public static LoginUser from(String username, String password, SysUser user) {
        return from(username, password, user, Collections.emptyList());
    }

    @Override
    public boolean isAccountNonExpired() {
        return expireTime > System.currentTimeMillis();
    }

    @Override
    public boolean isEnabled() {
        return user.getStatus();
    }
}
