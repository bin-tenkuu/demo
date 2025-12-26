package demo.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
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
@Entity
@Table(name = "sys_user_auth", uniqueConstraints = {
        @UniqueConstraint(columnNames = {"type", "user_id"})
})
public class SysUserAuth {
    @Schema(description = "登录账号")
    @Id
    @Column(name = "username")
    private String username;
    @Schema(description = "登陆类型")
    @Column(name = "type")
    @Enumerated(EnumType.STRING)
    private SysUserAuthType type;
    @Schema(description = "用户ID")
    @Column(name = "user_id")
    private Long userId;
    @Schema(description = "密码")
    @Column(name = "password")
    private String password;
}
