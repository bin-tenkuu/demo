package demo.core.autoconfigure.mybatis;

import com.baomidou.mybatisplus.core.handlers.MetaObjectHandler;
import lombok.RequiredArgsConstructor;
import org.apache.ibatis.reflection.MetaObject;

import java.util.List;

/// @author bin
/// @since 2025/12/15
@RequiredArgsConstructor
public class MetaObjectHandlers implements MetaObjectHandler {
    private final List<MetaObjectHandler> list;

    @Override
    public void insertFill(MetaObject metaObject) {
        for (var handler : list) {
            handler.insertFill(metaObject);
        }
    }

    @Override
    public void updateFill(MetaObject metaObject) {
        for (var handler : list) {
            handler.updateFill(metaObject);
        }
    }
}
