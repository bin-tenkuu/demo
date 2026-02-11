package demo.auth.entity;

import com.fasterxml.jackson.annotation.JsonIgnore;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;

/**
 * @author bin
 * @since 2025/07/15
 */
@Getter
@Setter
@Entity
@Table(name = "sys_user")
public class SysUser extends BaseSys {
    @Schema(description = "用户ID")
    @Id
    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    // @NotNull(message = "用户ID不能为空")
    private Long id;
    public static final String ID = "id";
    @Schema(description = "用户昵称")
    @Column(name = "nick_name")
    private String nickName;
    @Schema(description = "用户邮箱")
    @Column(name = "email")
    private String email;
    @Schema(description = "手机号码")
    @Column(name = "phone_number")
    private String phoneNumber;
    @Schema(description = "头像路径")
    @Column(name = "avatar")
    private String avatar;
    @Schema(description = "帐号状态（0正常 1停用）")
    @Column(name = "status")
    private Integer status;
    public static final String STATUS = "status";
    @Schema(description = "最后登录IP")
    @Column(name = "login_ip")
    private String loginIp;
    @Schema(description = "最后登录时间")
    @Column(name = "login_date")
    private LocalDateTime loginDate;

    @OneToMany(mappedBy = "user", fetch = FetchType.LAZY)
    @JsonIgnore
    private Set<SysUserRole> userRoles = new HashSet<>();
}
