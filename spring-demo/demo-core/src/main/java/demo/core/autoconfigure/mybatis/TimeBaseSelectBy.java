package demo.core.autoconfigure.mybatis;

import com.baomidou.mybatisplus.core.injector.AbstractMethod;
import com.baomidou.mybatisplus.core.metadata.TableInfo;
import org.apache.ibatis.mapping.MappedStatement;

import java.util.Objects;

/// @author bin
/// @since 2025/10/09
public class TimeBaseSelectBy extends AbstractMethod {
    private final String where;

    public TimeBaseSelectBy(String methodName, String where) {
        super(methodName);
        this.where = Objects.requireNonNull(where);
    }

    @Override
    public MappedStatement injectMappedStatement(Class<?> mapperClass, Class<?> modelClass, TableInfo tableInfo) {
        var tableName = tableInfo.getTableName();
        String sb = """
                <script>SELECT "%s", "%s"
                <foreach collection="fields" item="field">,${field}</foreach>
                FROM %s
                WHERE %s
                </script>""".formatted(TimeBase.TIME, TimeBase.ID, tableName, where);
        var sqlSource = languageDriver.createSqlSource(configuration,
                sb,
                modelClass
        );
        return this.addSelectMappedStatementForOther(
                mapperClass, this.methodName, sqlSource, modelClass
        );
    }
}
