package demo.autoconfigure.mybatis;

import com.baomidou.mybatisplus.core.mapper.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

/// @author bin
/// @since 2025/05/06
public interface TimeBaseMapper<T extends TimeBase> extends Mapper<T> {
    int insert(T entity);

    T findById(@Param(TimeBase.TIME) LocalDateTime time, @Param(TimeBase.ID) String id);

    int merge(@Param("e") T entity);

    List<T> listByTimeAndSns(
            @Param("time") LocalDateTime time,
            @Param("ids") List<String> ids,
            @Param("fields") List<String> fields
    );

    T getByTimeAndSn(
            @Param("time") LocalDateTime time,
            @Param("id") String id,
            @Param("fields") List<String> fields
    );

    List<T> listBySnAndTimeRange(
            @Param("id") String id,
            @Param("start") LocalDateTime start,
            @Param("end") LocalDateTime end,
            @Param("fields") List<String> fields
    );
}
