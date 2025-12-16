package demo.autoconfigure.mybatis;

import com.baomidou.mybatisplus.core.injector.AbstractMethod;
import com.baomidou.mybatisplus.core.injector.AbstractSqlInjector;
import com.baomidou.mybatisplus.core.injector.methods.Insert;
import com.baomidou.mybatisplus.core.metadata.TableInfo;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.ibatis.session.Configuration;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

/// @author bin
/// @since 2025/05/06
@Slf4j
public class TimeBaseSqlInjector extends AbstractSqlInjector {
    @AllArgsConstructor
    public enum Type {
        TimeAndId("""
                "%s" = #{time} and "%s" = #{id}"""
                .formatted(TimeBase.TIME, TimeBase.ID)),
        TimeAndIds("""
                "%s" = #{time}
                <foreach collection="ids" item="id" open="and "%s" in (" close=")" separator=",">#{id}</foreach>"""
                .formatted(TimeBase.TIME, TimeBase.ID)),
        TimeRangeAndId("""
                "%s" between #{start} and #{end} and "%s" = #{id}"""
                .formatted(TimeBase.TIME, TimeBase.ID));
        private final String sql;
    }

    @Override
    public List<AbstractMethod> getMethodList(Configuration configuration, Class<?> mapperClass, TableInfo tableInfo) {
        if (TimeBaseMapper.class.isAssignableFrom(mapperClass)) {
            var dbConfig = GlobalConfigUtils.getDbConfig(configuration);
            log.debug("TimeBaseMapper: {}", mapperClass.getName());
            var list = new ArrayList<AbstractMethod>();
            list.add(new Insert(dbConfig.isInsertIgnoreAutoIncrementColumn()));
            list.add(new TimeBaseMerge("merge"));
            list.add(new TimeBaseFindById());
            list.add(build(TimeBaseSelectBy::new, "getBy", Type.TimeAndId));
            list.add(build(TimeBaseSelectBy::new, "listBy", Type.TimeAndIds));
            list.add(build(TimeBaseSelectBy::new, "listBy", Type.TimeRangeAndId));
            list.add(build(TimeBaseDeleteBy::new, "deleteBy", Type.TimeAndId));
            list.add(build(TimeBaseDeleteBy::new, "deleteBy", Type.TimeAndIds));
            list.add(build(TimeBaseDeleteBy::new, "deleteBy", Type.TimeRangeAndId));
            return list;
        }
        return null;
    }

    private static AbstractMethod build(BiFunction<String, String, AbstractMethod> newer, String name, Type type) {
        return newer.apply(name + type.name(), type.sql);
    }


}
