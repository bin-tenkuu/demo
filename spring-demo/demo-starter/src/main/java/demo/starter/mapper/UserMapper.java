package demo.starter.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import demo.starter.entity.User;
import org.apache.ibatis.annotations.Mapper;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/11
 */
@Mapper
public interface UserMapper extends BaseMapper<User> {

    void initTable();
}
