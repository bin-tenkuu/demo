package demo.auth.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import demo.auth.entity.SysUserAuth;
import org.apache.ibatis.annotations.Mapper;

/// @author bin
/// @since 2025/07/15
@Mapper
public interface SysUserAuthMapper extends BaseMapper<SysUserAuth> {

    SysUserAuth findByUsername(String username);

    int countByUsername(String username);
}
