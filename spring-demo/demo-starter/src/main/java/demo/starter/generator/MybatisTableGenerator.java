package demo.starter.generator;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import demo.starter.entity.User;
import lombok.AllArgsConstructor;

import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Date;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/12/26
 */
@AllArgsConstructor
public class MybatisTableGenerator {
    private final TypeMap typeMap;

    public static void main(String[] args) {
        var a = new MybatisTableGenerator(MySql);
        System.out.println(a.generateTable(User.class));
    }

    private String generateTable(Class<?> clazz) {
        var sb = new StringBuilder();
        sb.append("create table if not exists ");
        // 获取表名
        var name = clazz.getAnnotation(TableName.class);
        if (name != null) {
            sb.append(name.value());
        } else {
            sb.append(toSnakeCase(clazz.getSimpleName()));
        }
        sb.append(" (\n");
        // 获取字段名
        for (Field field : clazz.getDeclaredFields()) {
            var sqlType = typeMap.toSqlType(field.getType());
            var tableId = field.getAnnotation(TableId.class);
            if (tableId != null) {
                sb.append("    ");
                if (tableId.value().isEmpty()) {
                    sb.append(toSnakeCase(field.getName()));
                } else {
                    sb.append(tableId.value());
                }
                sb.append(" ").append(sqlType).append(" not null");
                sb.append("\n").append("            primary key");
            } else {
                var tableField = field.getAnnotation(TableField.class);
                if (tableField != null) {
                    if (!tableField.exist()) {
                        continue;
                    }
                    sb.append("    ");
                    if (tableField.value().isEmpty()) {
                        sb.append(toSnakeCase(field.getName()));
                    } else {
                        sb.append(tableField.value());
                    }
                } else {
                    sb.append("    ");
                    sb.append(toSnakeCase(field.getName()));
                }
                sb.append(" ").append(sqlType);
            }
            sb.append(",\n");
        }
        sb.setLength(sb.length() - 2);
        sb.append("\n);");
        return sb.toString();
    }

    private String toSnakeCase(String name) {
        var sb = new StringBuilder(name.length());
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (Character.isUpperCase(c)) {
                sb.append("_");
                sb.append(Character.toLowerCase(c));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    @FunctionalInterface
    public interface TypeMap {
        String toSqlType(Class<?> clazz);
    }

    private static final TypeMap Sqlite = clazz -> {
        if (Integer.class.isAssignableFrom(clazz)
                || Long.class.isAssignableFrom(clazz)
                || Date.class.isAssignableFrom(clazz)
        ) {
            return "integer";
        } else if (String.class.isAssignableFrom(clazz)) {
            return "text";
        } else if (Float.class.isAssignableFrom(clazz)
                || Double.class.isAssignableFrom(clazz)
        ) {
            return "real";
        }
        return "text";
    };
    private static final TypeMap MySql = clazz -> {
        if (Byte.class.isAssignableFrom(clazz)) {
            return "tinyint";
        } else if (Short.class.isAssignableFrom(clazz)) {
            return "smallint";
        } else if (Integer.class.isAssignableFrom(clazz)) {
            return "integer";
        } else if (Long.class.isAssignableFrom(clazz)) {
            return "bigint";
        } else if (Float.class.isAssignableFrom(clazz)) {
            return "float";
        } else if (Double.class.isAssignableFrom(clazz)) {
            return "double";
        } else if (BigDecimal.class.isAssignableFrom(clazz)) {
            return "decimal(19,2)";
        } else if (java.sql.Date.class.isAssignableFrom(clazz)) {
            return "date";
        } else if (java.sql.Time.class.isAssignableFrom(clazz)) {
            return "time";
        } else if (java.sql.Timestamp.class.isAssignableFrom(clazz)) {
            return "timestamp";
        } else if (LocalDateTime.class.isAssignableFrom(clazz)) {
            return "datetime";
        } else if (String.class.isAssignableFrom(clazz)) {
            return "varchar(255)";
        }
        return "text";
    };
}
