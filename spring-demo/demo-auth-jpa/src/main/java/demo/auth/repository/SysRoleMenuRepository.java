package demo.auth.repository;

import demo.auth.entity.SysMenu;
import demo.auth.entity.SysRoleMenu;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.jpa.repository.support.JpaRepositoryImplementation;

import java.util.List;

/**
 * @author bin
 * @since 2025/12/26
 */
public interface SysRoleMenuRepository extends JpaRepositoryImplementation<SysRoleMenu, SysRoleMenu.Id> {

    @Query("delete from SysRoleMenu srm where srm.id.roleId in ?1")
    @Modifying
    void deleteByMenuIds(List<Long> ids);

    @Query("select rm.menu from SysRoleMenu rm where rm.id.roleId = ?1")
    List<SysMenu> findMenuByRoleId(Long roleId);

    void deleteByRoleId(Long roleId);

    void deleteByRoleIdAndMenuIdIn(@NotNull Long roleId, @NotEmpty List<Long> menuIds);
}
