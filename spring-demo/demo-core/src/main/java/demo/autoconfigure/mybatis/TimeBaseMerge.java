package demo.autoconfigure.mybatis;

import com.baomidou.mybatisplus.core.injector.AbstractMethod;
import com.baomidou.mybatisplus.core.metadata.TableInfo;
import org.apache.ibatis.mapping.MappedStatement;

/// @author bin
/// @since 2025/10/09
public class TimeBaseMerge extends AbstractMethod {

    protected TimeBaseMerge(String methodName) {
        super(methodName);
    }

    @Override
    public MappedStatement injectMappedStatement(Class<?> mapperClass, Class<?> modelClass, TableInfo tableInfo) {
        var columns = new StringBuilder(TimeBase.TIME + "," + TimeBase.ID);
        var values = new StringBuilder("union all select #{e.time},#{e.id}");
        var sets = new StringBuilder();
        var insert = new StringBuilder("n." + TimeBase.TIME + ",n." + TimeBase.ID);
        for (var info : tableInfo.getFieldList()) {
            var column = info.getColumn();
            if (TimeBase.TIME.equalsIgnoreCase(column) || TimeBase.ID.equalsIgnoreCase(column)) {
                continue;
            }
            columns.append(",").append(column);
            values.append(",#{e.").append(info.getProperty()).append("}");
            sets.append("t.").append(column)
                    .append("=IFNULL(n.").append(column)
                    .append(",t.").append(column).append("),\n");
            insert.append(",n.").append(column);
        }
        sets.setLength(sets.length() - 2);

        var tableName = tableInfo.getTableName();
        String sb = "merge into " + tableName + " t\n" +
                "using (select " + columns + " \n" +
                "from " + tableName + " where false \n" +
                values + " \n) n on t.time = n.time and t.id = n.id \n" +
                "when matched then update set \n" + sets + "\n" +
                "when not matched then insert(" + columns + ") \n" +
                "values(" + insert + ")";

        var sqlSource = languageDriver.createSqlSource(configuration,
                sb,
                modelClass
        );
        return this.addUpdateMappedStatement(mapperClass, modelClass, this.methodName, sqlSource);
    }
}
