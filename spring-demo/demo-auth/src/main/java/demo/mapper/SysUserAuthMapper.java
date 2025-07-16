package demo.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import demo.entity.SysUserAuth;
import org.apache.ibatis.annotations.Mapper;

/**
 * @author bin
 * @since 2025/07/15
 */
@Mapper
public interface SysUserAuthMapper extends BaseMapper<SysUserAuth> {

    void initTable();

    SysUserAuth findByUsername(String username);

}
