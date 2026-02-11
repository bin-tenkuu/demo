package demo.auth.mapper;


import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.Constants;
import demo.auth.entity.SysRole;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/// 角色信息表 Mapper 接口
///
/// @author bin
/// @since 2023-05-30 09:40:18
public interface SysRoleMapper extends BaseMapper<SysRole> {

    <P extends IPage<SysRole>> P selectPageWithMenu(P page, @Param("menuId") Long menuId,
            @Param(Constants.WRAPPER) Wrapper<SysRole> queryWrapper);

    List<SysRole> listByRoleKey(@Param("roleKey") String roleKey);
}
