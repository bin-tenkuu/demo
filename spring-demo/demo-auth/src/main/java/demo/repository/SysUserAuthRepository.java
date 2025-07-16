package demo.repository;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import demo.entity.SysUserAuth;
import demo.mapper.SysUserAuthMapper;
import org.springframework.stereotype.Service;

import java.util.HashMap;

/**
 * @author bin
 * @since 2025/07/15
 */
@Service
public class SysUserAuthRepository extends ServiceImpl<SysUserAuthMapper, SysUserAuth> {
    private static final HashMap<String, SysUserAuth> extryAuth = new HashMap<>();

    public static void addExtryAuth(SysUserAuth auth) {
        extryAuth.put(auth.getUsername(), auth);
    }

    public static void addExtryAuth(String username, String password, Long id) {
        SysUserAuth auth = new SysUserAuth();
        auth.setUsername(username);
        auth.setPassword("{noop}" + password);
        auth.setUserId(id);
        extryAuth.put(username, auth);
    }

    public static void removeExtryAuth(String username) {
        extryAuth.remove(username);
    }

    public SysUserAuth findByUsername(String username) {
        var auth = extryAuth.remove(username);
        if (auth != null) {
            return auth;
        }
        return baseMapper.findByUsername(username);
    }

}
