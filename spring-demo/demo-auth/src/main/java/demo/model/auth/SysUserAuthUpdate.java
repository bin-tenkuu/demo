package demo.model.auth;

import demo.entity.SysUserAuthType;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @since 2025/12/16
 */
@Getter
@Setter
@Schema(description = "系统用户认证信息修改")
public class SysUserAuthUpdate {
    @Schema(description = "登录账号")
    private String username;
    @Schema(description = "登陆类型")
    private SysUserAuthType type;
    @Schema(description = "新密码")
    private String oldPassword;
    @Schema(description = "新密码")
    private String newPassword;
}
