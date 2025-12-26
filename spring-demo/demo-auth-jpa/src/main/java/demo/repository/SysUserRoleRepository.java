package demo.repository;

import demo.entity.SysUserRole;
import org.springframework.data.jpa.repository.support.JpaRepositoryImplementation;

import java.util.Collection;
import java.util.List;

/**
 * @author bin
 * @since 2025/12/26
 */
public interface SysUserRoleRepository extends JpaRepositoryImplementation<SysUserRole, SysUserRole.Id> {
    void deleteByUserIdAndRoleIdIn(Long userId, List<Long> roleIds);

    void deleteByRoleIdAndUserIdIn(Long roleId, List<Long> userIds);

    void deleteByUserIdIn(List<Long> userId);

    void deleteByUserId(Long userId);

    void deleteByRoleIdIn(Collection<Long> roleIds);

}
