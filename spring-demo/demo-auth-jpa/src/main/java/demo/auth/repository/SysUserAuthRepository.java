package demo.auth.repository;

import demo.auth.entity.SysUserAuth;
import demo.core.util.CacheMap;
import org.springframework.data.jpa.repository.support.JpaRepositoryImplementation;

import java.time.Duration;

/**
 * @author bin
 * @since 2025/07/15
 */
public interface SysUserAuthRepository extends JpaRepositoryImplementation<SysUserAuth, String> {
    CacheMap<String, SysUserAuth> extryAuth = new CacheMap<>();

    SysUserAuth findByUsername(String username);

    int countByUsername(String username);

    default void addExtryAuth(SysUserAuth auth, Duration duration) {
        extryAuth.set(auth.getUsername(), auth, duration.toMillis());
    }

    default SysUserAuth getExtryAuth(String username) {
        return extryAuth.get(username);
    }

    default SysUserAuth findAllByUsername(String username) {
        var auth = extryAuth.remove(username);
        if (auth != null) {
            return auth;
        }
        return findByUsername(username);
    }

    default boolean checkUserNameExist(String username) {
        return countByUsername(username) > 0;
    }

}
