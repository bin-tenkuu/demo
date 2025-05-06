package demo.autoconfigure.mybatisSqlInject;

import com.baomidou.mybatisplus.core.injector.AbstractMethod;
import com.baomidou.mybatisplus.core.injector.DefaultSqlInjector;
import com.baomidou.mybatisplus.core.metadata.TableInfo;
import lombok.val;
import org.apache.ibatis.session.Configuration;

import java.util.List;

/**
 * @author bin
 * @since 2025/05/06
 */
public class TimeBaseSqlInjector extends DefaultSqlInjector {
    @Override
    public List<AbstractMethod> getMethodList(Configuration configuration, Class<?> mapperClass, TableInfo tableInfo) {
        val list = super.getMethodList(configuration, mapperClass, tableInfo);
        list.add(new TimeBaseFindById());
        return list;
    }
}
