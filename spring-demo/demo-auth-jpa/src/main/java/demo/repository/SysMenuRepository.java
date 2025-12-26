package demo.repository;


import demo.constant.UserConstants;
import demo.entity.SysMenu;
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

    @Query(value = """
            select *
            from (select distinct menu_id
                  from sys_user_role sur
                           inner join sys_role_menu srm on sur.role_id = srm.role_id
                where user_id = :userId
            ) sur
                inner join sys_menu sm on sm.id = sur.menu_id
            where sm.`status` = 0
            order by parent_id, order_num
            """, nativeQuery = true)
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
