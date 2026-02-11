package demo.auth.model.auth;

import demo.auth.entity.SysUser;
import lombok.Getter;
import lombok.Setter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.Collection;
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

    public void buildAuthorities(Collection<String> list) {
        authorities = list.stream().map(SimpleGrantedAuthority::new).collect(Collectors.toList());
    }

    public static LoginUser from(String username, String password, SysUser user) {
        var loginUser = new LoginUser();
        loginUser.setUsername(username);
        loginUser.setPassword(password);
        loginUser.setUser(user);
        loginUser.setAuthorities(List.of());
        loginUser.setLoginTime(System.currentTimeMillis());
        return loginUser;
    }

    @Override
    public boolean isAccountNonExpired() {
        return expireTime > System.currentTimeMillis();
    }

    @Override
    public boolean isEnabled() {
        return user.getStatus() == 0;
    }
}
