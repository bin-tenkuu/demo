package demo.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

/**
 * @author bin
 * @since 2025/07/15
 */
@Getter
@Setter
@ToString
@TableName(value = "sys_user_auth")
public class SysUserAuth {
    @Schema(description = "登录账号")
    @TableId("username")
    private String username;
    @Schema(description = "登陆类型")
    @TableField("type")
    private SysUserAuthType type;
    @Schema(description = "用户ID")
    @TableField("user_id")
    private Long userId;
    @Schema(description = "密码")
    @TableField("password")
    private String password;
}
