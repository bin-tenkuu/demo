package demo.model.auth;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonUnwrapped;
import demo.entity.SysMenu;
import lombok.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * @author bin
 * @since 2023/06/01
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class MenuSelect {

    @JsonUnwrapped
    private SysMenu menu;

    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private List<MenuSelect> children;

    public MenuSelect(SysMenu menu) {
        this.menu = menu;
    }

    public static List<MenuSelect> build(List<SysMenu> list) {
        val selectMap = new HashMap<Long, MenuSelect>(list.size() + 1);
        // 假定 list 已经排序，且根节点只有1个，则 size <= 1 （循环节点会被移除）
        val midList = new ArrayList<MenuSelect>();
        // 第一次循环，所有节点放入 selectMap 中，已经找到父节点的直接设置 children，未找到父节点的放入 midList 中
        for (val entity : list) {
            val select = new MenuSelect(entity);
            selectMap.put(entity.getId(), select);
            val parent = selectMap.get(entity.getParentId());
            if (parent != null) {
                parent.getChildren().add(select);
            } else {
                midList.add(select);
            }
        }
        // 第二次循环，已经找到父节点的从 trees 中移除，剩下的为根节点
        val iterator = midList.iterator();
        while (iterator.hasNext()) {
            val select = iterator.next();
            val parent = selectMap.get(select.getMenu().getParentId());
            if (parent != null) {
                parent.getChildren().add(select);
                iterator.remove();
            }
        }
        return new ArrayList<>(midList);
    }

}
