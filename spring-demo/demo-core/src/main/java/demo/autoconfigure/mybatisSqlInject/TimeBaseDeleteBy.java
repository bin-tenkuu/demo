package demo.autoconfigure.mybatisSqlInject;

import com.baomidou.mybatisplus.core.injector.AbstractMethod;
import com.baomidou.mybatisplus.core.metadata.TableInfo;
import org.apache.ibatis.mapping.MappedStatement;

import java.util.Objects;

/**
 * @author bin
 * @since 2025/10/09
 */
public class TimeBaseDeleteBy extends AbstractMethod {
    private final String where;

    public TimeBaseDeleteBy(String methodName, String where) {
        super(methodName);
        this.where = Objects.requireNonNull(where);
    }

    @Override
    public MappedStatement injectMappedStatement(Class<?> mapperClass, Class<?> modelClass, TableInfo tableInfo) {
        String sb = """
                <script>DELETE FROM %s WHERE %s
                </script>""".formatted(tableInfo.getTableName(), where);
        var sqlSource = languageDriver.createSqlSource(configuration,
                sb,
                modelClass
        );
        return this.addSelectMappedStatementForOther(
                mapperClass, this.methodName, sqlSource, modelClass
        );
    }
}
