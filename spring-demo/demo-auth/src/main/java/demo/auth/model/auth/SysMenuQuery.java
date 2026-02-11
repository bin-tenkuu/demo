package demo.auth.model.auth;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import demo.auth.entity.SysMenu;
import demo.core.model.Query;
import lombok.Getter;
import lombok.Setter;
import org.springframework.util.StringUtils;

/**
 * @author bin
 * @since 2023/05/30
 */
@Getter
@Setter
public class SysMenuQuery extends Query<SysMenu> {

    /**
     * 菜单名称
     */
    @TableField("menu_name")
    private String menuName;

    /**
     * 父菜单ID
     */
    @TableField("parent_id")
    private Long parentId;

    /**
     * 是否为外链（0是 1否）
     */
    @TableField("is_frame")
    private Integer isFrame;

    /**
     * 是否缓存（0缓存 1不缓存）
     */
    @TableField("is_cache")
    private Integer isCache;

    /**
     * 菜单类型（M目录 C菜单 F按钮）
     */
    @TableField("menu_type")
    private String menuType;

    /**
     * 菜单状态（0显示 1隐藏）
     */
    @TableField("visible")
    private Integer visible;

    /**
     * 菜单状态（0正常 1停用）
     */
    @TableField("status")
    private Integer status;

    /**
     * 权限标识
     */
    @TableField("perms")
    private String perms;

    @Override
    public QueryWrapper<SysMenu> toQuery() {
        return buildQuery()
                .like(StringUtils.hasLength(menuName), SysMenu.MENU_NAME, menuName)
                .eq(parentId != null, SysMenu.PARENT_ID, parentId)
                .eq(isFrame != null, SysMenu.IS_FRAME, isFrame)
                .eq(isCache != null, SysMenu.IS_CACHE, isCache)
                .eq(StringUtils.hasLength(menuType), SysMenu.MENU_TYPE, menuType)
                .eq(visible != null, SysMenu.VISIBLE, visible)
                .eq(status != null, SysMenu.STATUS, status)
                .like(StringUtils.hasLength(perms), SysMenu.PERMS, perms)
                .orderByAsc(SysMenu.PARENT_ID, SysMenu.ORDER_NUM)
                ;
    }
}
