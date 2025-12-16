package demo.autoconfigure.mybatis;

import com.baomidou.mybatisplus.core.batch.MybatisBatch;
import com.baomidou.mybatisplus.core.toolkit.MybatisBatchUtils;
import com.baomidou.mybatisplus.core.toolkit.reflect.GenericTypeUtils;
import lombok.Getter;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.beans.factory.annotation.Autowired;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/// @author bin
/// @since 2025/04/10
@SuppressWarnings("unused")
public abstract class TimeBaseService<M extends TimeBaseMapper<T>, T extends TimeBase> {
    @Getter
    @Autowired
    protected M baseMapper;
    @Getter
    @Autowired
    protected SqlSessionFactory sqlSessionFactory;

    protected final Class<?>[] typeArguments = GenericTypeUtils.resolveTypeArguments(getClass(), TimeBaseService.class);

    @SuppressWarnings("unchecked")
    protected Class<M> getBaseMapperClass() {
        return (Class<M>) typeArguments[0];
    }

    private static <K, V> Map<K, V> toMap(List<V> list, Function<V, K> mapper) {
        var map = new HashMap<K, V>();
        for (var v : list) {
            map.put(mapper.apply(v), v);
        }
        return map;
    }

    public void merge(T entity) {
        baseMapper.merge(entity);
    }

    public void merge(List<T> entityList) {
        var method = new MybatisBatch.Method<T>(getBaseMapperClass());
        var batchMethod = method.<T>get("merge", (user) -> Map.of("e", user));
        MybatisBatchUtils.execute(sqlSessionFactory, entityList, true, batchMethod);
    }

    public Map<String, T> listByTimeAndSns(
            LocalDateTime time,
            List<String> sns,
            String... fields
    ) {
        var list = baseMapper.listByTimeAndSns(time, sns, Arrays.asList(fields));
        return toMap(list, TimeBase::getId);
    }

    public T getByTimeAndSn(
            LocalDateTime time,
            String sn,
            String... fields
    ) {
        return baseMapper.getByTimeAndSn(time, sn, Arrays.asList(fields));
    }

    public Map<LocalDateTime, T> listBySnAndTimeRange(
            String sn,
            LocalDateTime start,
            LocalDateTime end,
            String... fields
    ) {
        var list = baseMapper.listBySnAndTimeRange(sn, start, end, Arrays.asList(fields));
        return toMap(list, TimeBase::getTime);
    }

}
