package demo.mapper;

import demo.autoconfigure.mybatisSqlInject.TimeBaseMapper;
import demo.entity.UserData;
import org.apache.ibatis.annotations.Mapper;

/**
 * @author bin
 * @since 2025/05/06
 */
@Mapper
public interface UserDataMapper extends TimeBaseMapper<UserData> {

    void initTable();
}
