package demo.mapper;

import demo.entity.SysMenu;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/// 角色与菜单关联表 数据层
///
/// @author ruoyi
public interface SysRoleMenuMapper {

    int insertRoleMenu(@Param("roleId") Long roleId, @Param("menuIds") List<Long> menuIds);

    int deleteByRoleMenu(@Param("roleId") Long roleId, @Param("menuIds") List<Long> menuIds);

    void deleteByMenuIds(@Param("ids") List<Long> ids);

    void deleteByRoleId(Long id);

    List<SysMenu> selectMenuByRoleId(@Param("roleId") Long roleId);
}
