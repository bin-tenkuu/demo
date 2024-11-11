package demo.entity;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/11
 */
@Data
@TableName("user")
public class User {
    @TableId
    private Integer id;
    private String name;
}
