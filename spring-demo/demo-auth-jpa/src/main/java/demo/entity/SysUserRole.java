package demo.entity;

import jakarta.persistence.*;
import lombok.*;

/**
 * @author bin
 * @since 2025/12/26
 */
@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "sys_user_role")
public class SysUserRole {
    @EmbeddedId
    private Id id;

    @ManyToOne(fetch = FetchType.LAZY)
    @MapsId("userId")
    @JoinColumn(name = "user_id", insertable = false, updatable = false)
    private SysUser user;

    @ManyToOne(fetch = FetchType.LAZY)
    @MapsId("roleId")
    @JoinColumn(name = "role_id", insertable = false, updatable = false)
    private SysRole role;

    public SysUserRole(Long userId, Long roleId) {
        this.id = new Id(userId, roleId);
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Embeddable
    public static class Id {
        @Column(name = "user_id")
        private Long userId;
        @Column(name = "role_id")
        private Long roleId;
    }
}
