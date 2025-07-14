package demo.autoconfigure.mybatisSqlInject;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;

/**
 * @author bin
 * @since 2025/05/06
 */
public interface TimeBaseMapper<T extends TimeBase> extends BaseMapper<T> {
    T findById(@Param(TimeBaseFindById.TIME) LocalDateTime time, @Param(TimeBaseFindById.ID) String id);
}
