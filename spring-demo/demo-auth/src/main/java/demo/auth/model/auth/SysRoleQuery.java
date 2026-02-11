package demo.auth.model.auth;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import demo.auth.entity.SysRole;
import demo.core.model.Query;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;

/**
 * @author bin
 * @since 2023/05/30
 */
@Getter
@Setter
public class SysRoleQuery extends Query<SysRole> {

    /**
     * 角色名称
     */
    @Schema(description = "角色名称")
    private String roleName;

    /**
     * 角色权限字符串
     */
    @Schema(description = "角色权限字符串")
    private String roleKey;

    /**
     * 角色状态（0正常 1停用）
     */
    @Schema(description = "角色状态（0正常 1停用）")
    private String status;

    /**
     * 创建时间
     */
    @Schema(description = "创建时间")
    private LocalDateTime createTimeStart;
    private LocalDateTime createTimeEnd;

    /**
     * 更新时间
     */
    @Schema(description = "更新时间")
    private LocalDateTime updateTimeStart;
    private LocalDateTime updateTimeEnd;


    @Override
    public QueryWrapper<SysRole> toQuery() {
        return buildQuery()
                .like(StringUtils.hasText(roleName), "role_name", roleName)
                .like(StringUtils.hasText(roleKey), "role_key", roleKey)
                .eq(StringUtils.hasText(status), "status", status)
                .ge(createTimeStart != null, "create_time", createTimeStart)
                .le(createTimeEnd != null, "create_time", createTimeEnd)
                .ge(updateTimeStart != null, "update_time", updateTimeStart)
                .le(updateTimeEnd != null, "update_time", updateTimeEnd)
                ;
    }
}
