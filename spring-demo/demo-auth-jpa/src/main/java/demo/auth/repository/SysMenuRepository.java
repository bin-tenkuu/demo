package demo.auth.repository;


import demo.auth.constant.UserConstants;
import demo.auth.entity.SysMenu;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.jpa.repository.support.JpaRepositoryImplementation;
import org.springframework.data.repository.query.Param;

import java.util.List;

/**
 * <p>
 * 菜单权限表 服务实现类
 * </p>
 *
 * @author bin
 * @since 2023-05-30 09:40:18
 */
public interface SysMenuRepository extends JpaRepositoryImplementation<SysMenu, Long> {

    @Query("select m from SysMenu m where m.menuName = :name")
    List<SysMenu> listByMenuName(String name);

    @Query("""
            select sm
            from SysMenu sm
                inner join (
                    select distinct srm.id.menuId as menuId
                    from SysUserRole sur
                        inner join SysRoleMenu srm on sur.id.roleId = srm.id.roleId
                    where sur.id.userId = :userId
                ) srm on sm.id = srm.menuId
            where sm.status = 0
            order by sm.parentId, sm.orderNum
            """)
    List<SysMenu> listByUserId(@Param("userId") Long userId);

    @Query("select m from SysMenu m order by m.parentId asc, m.orderNum asc")
    List<SysMenu> listAll();

    /**
     * 校验菜单名称是否唯一
     *
     * @param menu 菜单信息
     * @return 结果
     */
    default boolean checkMenuNameUnique(SysMenu menu) {
        var menus = listByMenuName(menu.getMenuName());
        var id = menu.getId();
        for (var sysMenu : menus) {
            if (id == null) {
                break;
            }
            if (!sysMenu.getId().equals(id)) {
                return true;
            }
        }
        return false;
    }

    default List<SysMenu> listMenuByUserId(Long userId) {
        if (userId == null || userId.equals(UserConstants.ADMIN_ID)) {
            // 管理员拥有所有权限
            return listAll();
        }
        return listByUserId(userId);
    }

}
