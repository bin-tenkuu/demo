package demo.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import demo.entity.SysUser;
import org.apache.ibatis.annotations.Mapper;

/**
 * @author bin
 * @since 2025/07/15
 */
@Mapper
public interface SysUserMapper extends BaseMapper<SysUser> {

    void initTable();

}
