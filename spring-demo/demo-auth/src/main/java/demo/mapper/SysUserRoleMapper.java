package demo.mapper;

import demo.entity.SysMenu;
import demo.entity.SysRole;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/// 用户与角色关联表 数据层
///
/// @author ruoyi
public interface SysUserRoleMapper {
    int insertUserRoles(@Param("userId") Long userId, @Param("roleIds") List<Long> roleIds);

    int insertUsersRole(@Param("roleId") Long roleId, @Param("userIds") List<Long> userIds);

    int deleteByUserRoles(@Param("userId") Long userId, @Param("roleIds") List<Long> roleIds);

    int deleteByUsersRole(@Param("roleId") Long roleId, @Param("userIds") List<Long> userIds);

    void deleteByRoleId(@Param("ids") List<Long> ids);

    void deleteByUserId(Long id);

    List<SysRole> selectRoleByUserId(@Param("userId") Long userId);

    List<SysMenu> selectMenuByUserId(@Param("userId") Long userId);
}
