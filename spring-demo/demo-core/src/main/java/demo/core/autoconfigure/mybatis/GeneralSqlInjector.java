package demo.core.autoconfigure.mybatis;

import com.baomidou.mybatisplus.core.injector.AbstractMethod;
import com.baomidou.mybatisplus.core.injector.AbstractSqlInjector;
import com.baomidou.mybatisplus.core.injector.DefaultSqlInjector;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.TableInfo;
import lombok.AllArgsConstructor;
import org.apache.ibatis.session.Configuration;

import java.util.ArrayList;
import java.util.List;

/// @author bin
/// @since 2025/12/15
@AllArgsConstructor
public class GeneralSqlInjector extends AbstractSqlInjector {
    private final List<AbstractSqlInjector> injectors;
    private final DefaultSqlInjector defaultSqlInjector = new DefaultSqlInjector();

    @Override
    public List<AbstractMethod> getMethodList(Configuration configuration, Class<?> mapperClass, TableInfo tableInfo) {
        var list = new ArrayList<AbstractMethod>();
        if (BaseMapper.class.isAssignableFrom(mapperClass)) {
            var tmp = defaultSqlInjector.getMethodList(configuration, mapperClass, tableInfo);
            list.addAll(tmp);
        }
        for (var injector : injectors) {
            var tmp = injector.getMethodList(configuration, mapperClass, tableInfo);
            if (tmp == null || tmp.isEmpty()) {
                continue;
            }
            list.addAll(tmp);
        }
        return list;
    }
}
