package demo.model.auth;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

/// @author bin
/// @since 2023/05/30
@Getter
@Setter
@Schema(description = "系统用户新建")
public class SysUserNewVo {

    @Schema(description = "登录账号")
    @NotNull(message = "登录账号不能为空")
    private String username;

    @Schema(description = "用户昵称")
    @NotNull(message = "用户昵称不能为空")
    private String nickName;

    @Schema(description = "用户邮箱")
    private String email;

    @Schema(description = "手机号码")
    private String phoneNumber;

    @Schema(description = "头像路径")
    private String avatar;

    @Schema(description = "密码")
    @NotNull(message = "密码不能为空")
    private String password;

    @Schema(description = "帐号状态（0正常 1停用）")
    @NotNull(message = "帐号状态不能为空")
    private Integer status;

    @Schema(description = "角色ids")
    private List<Long> roleIds;
}
