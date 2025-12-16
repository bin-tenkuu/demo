package demo.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonUnwrapped;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;


@SuppressWarnings("unused")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "通用树")
@JsonInclude(JsonInclude.Include.NON_NULL)
public class TreeModel {
    /// 节点中的数据
    @JsonUnwrapped
    private Object data;
    /// 子节点
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private List<TreeModel> children;

    public TreeModel(Object data) {
        this.data = data;
    }

    public static <T, ID> List<TreeModel> build(List<T> list, Function<T, ID> getId, Function<T, ID> getParentId) {
        // id -> model
        var selectMap = new HashMap<ID, TreeModel>(list.size() + 1);
        // 假定 list 已经排序，且根节点只有1个，则 size <= 1 （循环节点会被移除）
        // parentId -> children
        var midMap = new HashMap<ID, ArrayList<TreeModel>>();
        // 所有节点放入 selectMap 中，已经找到父节点的直接设置 children，未找到父节点的放入 midMap 中等待寻找
        for (var entity : list) {
            var id = getId.apply(entity);
            var children = midMap.remove(id);
            var select = new TreeModel(entity, children);
            selectMap.put(id, select);
            var parentId = getParentId.apply(entity);
            var parent = selectMap.get(parentId);
            if (parent != null) {
                parent.getChildren().add(select);
            } else {
                midMap.computeIfAbsent(parentId, k -> new ArrayList<>()).add(select);
            }
        }
        // 剩下的为根节点
        var rootList = new ArrayList<TreeModel>();
        for (ArrayList<TreeModel> value : midMap.values()) {
            rootList.addAll(value);
        }
        return rootList;
    }
}
