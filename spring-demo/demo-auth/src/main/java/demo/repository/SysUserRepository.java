package demo.repository;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import demo.entity.SysUser;
import demo.mapper.SysUserMapper;
import demo.model.LoginUser;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

/**
 * @author bin
 * @since 2025/07/15
 */
@Service
@RequiredArgsConstructor
public class SysUserRepository extends ServiceImpl<SysUserMapper, SysUser>
        implements InitializingBean, UserDetailsService {
    private final SysUserAuthRepository sysUserAuthRepository;

    @Override
    public void afterPropertiesSet() {
        baseMapper.initTable();
    }

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        val userAuth = sysUserAuthRepository.findByUsername(username);
        if (userAuth == null) {
            throw new UsernameNotFoundException("User not found: " + username);
        }
        val user = this.getById(userAuth.getUserId());

        return LoginUser.from(userAuth.getUsername(), userAuth.getPassword(), user);
    }

}
