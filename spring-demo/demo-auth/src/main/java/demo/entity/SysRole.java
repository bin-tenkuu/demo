package demo.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @since 2025/12/15
 */
@Getter
@Setter
@TableName(value = "sys_role")
public class SysRole extends BaseSys {
    @Schema(description = "角色序号")
    @TableId("id")
    private Long id;
    public static final String ID = "id";
    @Schema(description = "角色权限")
    @TableField("role_key")
    private String roleKey;
    public static final String ROLE_KEY = "role_key";
    @Schema(description = "角色名称")
    @TableField("role_name")
    private String roleName;
    public static final String ROLE_NAME = "role_name";
    @Schema(description = "角色排序")
    @TableField("role_sort")
    private Integer roleSort;
    public static final String ROLE_SORT = "role_sort";
    @Schema(description = "角色状态")
    @TableField("status")
    private Integer status;
    public static final String STATUS = "status";

}
