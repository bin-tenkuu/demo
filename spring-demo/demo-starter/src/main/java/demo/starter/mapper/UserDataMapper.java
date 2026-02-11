package demo.starter.mapper;

import demo.core.autoconfigure.mybatis.TimeBaseMapper;
import demo.starter.entity.UserData;
import org.apache.ibatis.annotations.Mapper;

/**
 * @author bin
 * @since 2025/05/06
 */
@Mapper
public interface UserDataMapper extends TimeBaseMapper<UserData> {

    void initTable();
}
