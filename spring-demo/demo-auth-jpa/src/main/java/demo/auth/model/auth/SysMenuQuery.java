package demo.auth.model.auth;

import demo.auth.entity.SysMenu;
import demo.auth.model.JpaQuery;
import jakarta.persistence.criteria.CriteriaBuilder;
import jakarta.persistence.criteria.Predicate;
import jakarta.persistence.criteria.Root;
import lombok.Getter;
import lombok.Setter;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * @author bin
 * @since 2023/05/30
 */
@Getter
@Setter
public class SysMenuQuery extends JpaQuery<SysMenu> {
    public static final Specification<SysMenu> orderByAsc = (root, query, cb) -> {
        query.orderBy(cb.asc(root.get("parentId")), cb.asc(root.get("orderNum")));
        return query.getRestriction();
    };

    /**
     * 菜单名称
     */
    private String menuName;

    /**
     * 父菜单ID
     */
    private Long parentId;

    /**
     * 是否为外链（0是 1否）
     */
    private Integer isFrame;

    /**
     * 是否缓存（0缓存 1不缓存）
     */
    private Integer isCache;

    /**
     * 菜单类型（M目录 C菜单 F按钮）
     */
    private String menuType;

    /**
     * 菜单状态（0显示 1隐藏）
     */
    private Integer visible;

    /**
     * 菜单状态（0正常 1停用）
     */
    private Integer status;

    /**
     * 权限标识
     */
    private String perms;

    @Override
    public List<Predicate> toPredicate(Root<SysMenu> root, CriteriaBuilder cb) {
        var list = new ArrayList<Predicate>();
        if (StringUtils.hasLength(menuName)) {
            list.add(cb.like(root.get("menuName"), "%" + menuName + "%"));
        }
        if (parentId != null) {
            list.add(cb.equal(root.get("parentId"), parentId));
        }
        if (isFrame != null) {
            list.add(cb.equal(root.get("isFrame"), isFrame));
        }
        if (isCache != null) {
            list.add(cb.equal(root.get("isCache"), isCache));
        }
        if (StringUtils.hasLength(menuType)) {
            list.add(cb.equal(root.get("menuType"), menuType));
        }
        if (visible != null) {
            list.add(cb.equal(root.get("visible"), visible));
        }
        if (status != null) {
            list.add(cb.equal(root.get("status"), status));
        }
        if (StringUtils.hasLength(perms)) {
            list.add(cb.like(root.get("perms"), "%" + perms + "%"));
        }
        return list;
    }

}
