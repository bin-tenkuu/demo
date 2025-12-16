package demo.mapper;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.Constants;
import demo.entity.SysUser;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

/// @author bin
/// @since 2025/07/15
@Mapper
public interface SysUserMapper extends BaseMapper<SysUser> {

    <P extends IPage<SysUser>> P selectPageWithRole(
            P page,
            @Param("roleId") Long roleId,
            @Param(Constants.WRAPPER) Wrapper<SysUser> queryWrapper
    );

    <P extends IPage<SysUser>> P selectPageWithoutRole(
            P page,
            @Param(Constants.WRAPPER) Wrapper<SysUser> queryWrapper
    );
}
