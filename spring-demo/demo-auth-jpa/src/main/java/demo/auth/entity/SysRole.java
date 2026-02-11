package demo.auth.entity;

import com.fasterxml.jackson.annotation.JsonIgnore;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.util.HashSet;
import java.util.Set;

/**
 * @author bin
 * @since 2025/12/15
 */
@Getter
@Setter
@Entity
@Table(name = "sys_role")
public class SysRole extends BaseSys {
    @Schema(description = "角色序号")
    @Id
    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    public static final String ID = "id";
    @Schema(description = "角色权限")
    @Column(name = "role_key")
    private String roleKey;
    public static final String ROLE_KEY = "role_key";
    @Schema(description = "角色名称")
    @Column(name = "role_name")
    private String roleName;
    public static final String ROLE_NAME = "role_name";
    @Schema(description = "角色排序")
    @Column(name = "role_sort")
    private Integer roleSort;
    public static final String ROLE_SORT = "role_sort";
    @Schema(description = "角色状态")
    @Column(name = "status")
    private Integer status;
    public static final String STATUS = "status";

    @OneToMany(mappedBy = "role")
    @JsonIgnore
    private Set<SysUserRole> userRoles = new HashSet<>();
    @OneToMany(mappedBy = "role")
    @JsonIgnore
    private Set<SysRoleMenu> roleMenus = new HashSet<>();
}
