package demo.auth.model.auth;

import demo.auth.entity.SysRole;
import demo.auth.model.JpaQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.Predicate;
import jakarta.persistence.criteria.Root;
import lombok.Getter;
import lombok.Setter;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.List;

/**
 * @author bin
 * @since 2023/05/30
 */
@Getter
@Setter
public class SysRoleQuery extends JpaQuery<SysRole> {

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
    public List<Predicate> toPredicate(Root<SysRole> root, CriteriaBuilder cb) {
        var list = new java.util.ArrayList<Predicate>();
        if (StringUtils.hasText(roleName)) {
            list.add(cb.like(root.get("roleName"), "%" + roleName + "%"));
        }
        if (StringUtils.hasText(roleKey)) {
            list.add(cb.like(root.get("roleKey"), "%" + roleKey + "%"));
        }
        if (StringUtils.hasText(status)) {
            list.add(cb.equal(root.get("status"), status));
        }
        if (createTimeStart != null) {
            list.add(cb.greaterThanOrEqualTo(root.get("createTime"), createTimeStart));
        }
        if (createTimeEnd != null) {
            list.add(cb.lessThanOrEqualTo(root.get("createTime"), createTimeEnd));
        }
        if (updateTimeStart != null) {
            list.add(cb.greaterThanOrEqualTo(root.get("updateTime"), updateTimeStart));
        }
        if (updateTimeEnd != null) {
            list.add(cb.lessThanOrEqualTo(root.get("updateTime"), updateTimeEnd));
        }
        return list;
    }

}
