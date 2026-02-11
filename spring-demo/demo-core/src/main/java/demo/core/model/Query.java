package demo.core.model;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import lombok.Getter;
import lombok.Setter;

/// @author bin
/// @since 2025/12/15
@Getter
@Setter
public abstract class Query<T> {
    private int page = 1;
    private int size = 10;

    public Page<T> toPage() {
        return new Page<>(page, size);
    }

    public abstract QueryWrapper<T> toQuery();

    protected QueryWrapper<T> buildQuery() {
        return new QueryWrapper<>();
    }
}
