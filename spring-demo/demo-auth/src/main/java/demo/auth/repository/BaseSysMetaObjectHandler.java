package demo.auth.repository;

import com.baomidou.mybatisplus.core.handlers.MetaObjectHandler;
import demo.auth.entity.BaseSys;
import demo.auth.util.SecurityUtils;
import lombok.extern.slf4j.Slf4j;
import org.apache.ibatis.reflection.MetaObject;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * @author bin
 * @since 2025/12/15
 */
@Slf4j
@Component
public class BaseSysMetaObjectHandler implements MetaObjectHandler {
    @Override
    public void insertFill(MetaObject metaObject) {
        var username = SecurityUtils.getUsername().orElse(null);
        var entity = metaObject.getOriginalObject();
        if (entity instanceof BaseSys sys) {
            if (sys.getCreateBy() == null) {
                sys.setCreateBy(username);
            }
            if (sys.getCreateTime() == null) {
                sys.setCreateTime(LocalDateTime.now());
            }
            if (sys.getUpdateBy() == null) {
                sys.setUpdateBy(username);
            }
            if (sys.getUpdateTime() == null) {
                sys.setUpdateTime(LocalDateTime.now());
            }
        }
    }

    @Override
    public void updateFill(MetaObject metaObject) {
        var username = SecurityUtils.getUsername().orElse(null);
        var entity = metaObject.getOriginalObject();
        if (entity instanceof BaseSys sys) {
            if (sys.getUpdateBy() == null) {
                sys.setUpdateBy(username);
            }
            if (sys.getUpdateTime() == null) {
                sys.setUpdateTime(LocalDateTime.now());
            }
        }
    }
}
