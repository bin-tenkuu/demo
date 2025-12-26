package demo.entity;

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
@Table(name = "sys_menu")
public class SysMenu extends BaseSys {
    @Schema(description = "菜单ID")
    @Id
    @Column(name = "id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    public static final String ID = "id";
    @Schema(description = "菜单名称")
    @Column(name = "menu_name")
    private String menuName;
    public static final String MENU_NAME = "menu_name";
    @Schema(description = "父菜单ID")
    @Column(name = "parent_id")
    private Long parentId;
    public static final String PARENT_ID = "parent_id";
    @Schema(description = "显示顺序")
    @Column(name = "order_num")
    private Integer orderNum;
    public static final String ORDER_NUM = "order_num";
    @Schema(description = "路由地址")
    @Column(name = "path")
    private String path;
    public static final String PATH = "path";
    @Schema(description = "组件路径")
    @Column(name = "component")
    private String component;
    public static final String COMPONENT = "component";
    @Schema(description = "路由参数")
    @Column(name = "query")
    private String query;
    public static final String QUERY = "query";
    @Schema(description = "是否为外链（0是 1否）")
    @Column(name = "is_frame")
    private Integer isFrame;
    public static final String IS_FRAME = "is_frame";
    @Schema(description = "是否缓存（0缓存 1不缓存）")
    @Column(name = "is_cache")
    private Integer isCache;
    public static final String IS_CACHE = "is_cache";
    @Schema(description = "菜单类型（M目录 C菜单 F按钮）")
    @Column(name = "menu_type")
    private String menuType;
    public static final String MENU_TYPE = "menu_type";
    @Schema(description = "菜单状态（0显示 1隐藏）")
    @Column(name = "visible")
    private Integer visible;
    public static final String VISIBLE = "visible";
    @Schema(description = "菜单状态（0正常 1停用）")
    @Column(name = "status")
    private Integer status;
    public static final String STATUS = "status";
    @Schema(description = "权限标识")
    @Column(name = "perms")
    private String perms;
    public static final String PERMS = "perms";
    @Schema(description = "菜单图标")
    @Column(name = "icon")
    private String icon;
    public static final String ICON = "icon";

    @OneToMany(mappedBy = "menu")
    @JsonIgnore
    private Set<SysRoleMenu> roleMenus = new HashSet<>();
}
