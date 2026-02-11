package demo.core.autoconfigure.mybatis;

import com.baomidou.mybatisplus.core.injector.AbstractMethod;
import com.baomidou.mybatisplus.core.metadata.TableInfo;
import org.apache.ibatis.mapping.MappedStatement;

/// @author bin
/// @since 2025/05/06
public class TimeBaseFindById extends AbstractMethod {

    protected TimeBaseFindById() {
        super("findById");
    }

    @Override
    public MappedStatement injectMappedStatement(
            Class<?> mapperClass,
            Class<?> modelClass,
            TableInfo tableInfo
    ) {
        var sql = "SELECT * FROM " + tableInfo.getTableName() + " WHERE " +
                  TimeBase.TIME + " = #{" + TimeBase.TIME + "} AND " +
                  TimeBase.ID + " = #{" + TimeBase.ID + "}";
        var sqlSource = createSqlSource(
                configuration,
                "<script>" + sql + "</script>",
                modelClass
        );
        return addSelectMappedStatementForTable(mapperClass, sqlSource, tableInfo);
    }
}
