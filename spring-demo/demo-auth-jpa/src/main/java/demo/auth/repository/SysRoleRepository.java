package demo.auth.repository;


import demo.auth.entity.SysRole;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.jpa.repository.support.JpaRepositoryImplementation;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;

/// 角色信息表 服务实现类
///
/// @author bin
/// @since 2023-05-30 09:40:18
public interface SysRoleRepository extends JpaRepositoryImplementation<SysRole, Long> {

    @Query("select r from SysRole r where r.roleKey = :roleKey")
    List<SysRole> listByRoleKey(String roleKey);

    /// 校验角色权限是否唯一
    default boolean checkRoleKeyExist(SysRole role) {
        var roles = listByRoleKey(role.getRoleKey());
        var id = role.getId();
        for (var sysRole : roles) {
            if (id == null) {
                break;
            }
            if (!sysRole.getId().equals(id)) {
                return true;
            }
        }
        return false;
    }

    List<Long> findIdByStatusAndIdIn(Integer status, Collection<Long> ids);

    @Query("update SysRole r set r.status = 1, r.updateBy = :updateBy where r.id in :ids")
    @Modifying
    void updateAllByIdIn(String updateBy, HashSet<Long> ids);
}
