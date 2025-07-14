package demo.autoconfigure.mybatisSqlInject;

import com.baomidou.mybatisplus.core.injector.AbstractMethod;
import com.baomidou.mybatisplus.core.metadata.TableInfo;
import lombok.val;
import org.apache.ibatis.mapping.MappedStatement;

/**
 * @author bin
 * @since 2025/05/06
 */
public class TimeBaseFindById extends AbstractMethod {
    public static final String TIME = "TIME";
    public static final String ID = "ID";

    protected TimeBaseFindById() {
        super("findById");
    }

    @Override
    public MappedStatement injectMappedStatement(
            Class<?> mapperClass,
            Class<?> modelClass,
            TableInfo tableInfo
    ) {
        val sql = "SELECT * FROM " + tableInfo.getTableName() + " WHERE " +
                  TIME + " = #{" + TIME + "} AND " +
                  ID + " = #{" + ID + "}";
        val sqlSource = createSqlSource(
                configuration,
                "<script>" + sql + "</script>",
                modelClass
        );
        return addSelectMappedStatementForTable(mapperClass, sqlSource, tableInfo);
    }
}
