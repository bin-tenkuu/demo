package demo.auth.model.auth;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

/**
 * @author bin
 * @since 2023/05/30
 */
@Getter
@Setter
@Schema(description = "系统角色新增")
public class SysRoleNewVo {

    @Schema(description = "角色权限字符串")
    @NotBlank(message = "权限字符不能为空")
    private String roleKey;

    @Schema(description = "角色名称")
    @NotBlank(message = "角色名称不能为空")
    private String roleName;

    @Schema(description = "显示顺序")
    @NotNull(message = "显示顺序不能为空")
    private Integer roleSort;

    @Schema(description = "角色状态（0正常 1停用）")
    @NotNull(message = "角色状态不能为空")
    private Integer status;

    @Schema(description = "菜单ids")
    private List<Long> menuIds;
}
