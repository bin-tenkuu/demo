package demo.auth.repository;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import demo.auth.entity.SysUserAuth;
import demo.auth.mapper.SysUserAuthMapper;
import demo.core.util.CacheMap;
import org.springframework.stereotype.Service;

import java.time.Duration;

/**
 * @author bin
 * @since 2025/07/15
 */
@Service
public class SysUserAuthRepository extends ServiceImpl<SysUserAuthMapper, SysUserAuth> {
    private static final CacheMap<String, SysUserAuth> extryAuth = new CacheMap<>();

    public void addExtryAuth(SysUserAuth auth, Duration duration) {
        extryAuth.set(auth.getUsername(), auth, duration.toMillis());
    }

    public SysUserAuth getExtryAuth(String username) {
        return extryAuth.get(username);
    }

    public SysUserAuth findByUsername(String username) {
        var auth = extryAuth.remove(username);
        if (auth != null) {
            return auth;
        }
        return baseMapper.findByUsername(username);
    }

    public boolean checkUserNameExist(String username) {
        return baseMapper.countByUsername(username) > 0;
    }

}
