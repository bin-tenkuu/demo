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
@Table(name = "sys_role_menu")
public class SysRoleMenu {
    @EmbeddedId
    private Id id;

    @ManyToOne(fetch = FetchType.LAZY)
    @MapsId("roleId")
    @JoinColumn(name = "role_id", insertable = false, updatable = false)
    private SysRole role;

    @ManyToOne(fetch = FetchType.LAZY)
    @MapsId("menuId")
    @JoinColumn(name = "menu_id", insertable = false, updatable = false)
    private SysMenu menu;

    public SysRoleMenu(Long roleId, Long menuId) {
        this.id = new Id(roleId, menuId);
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Embeddable
    public static class Id {
        @Column(name = "role_id")
        private Long roleId;
        @Column(name = "menu_id")
        private Long menuId;
    }
}
