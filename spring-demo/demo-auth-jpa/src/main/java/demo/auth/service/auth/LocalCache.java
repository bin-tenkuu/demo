package demo.auth.service.auth;

import demo.auth.model.auth.LoginUser;
import demo.core.util.CacheMap;
import org.springframework.stereotype.Component;

/**
 * spring redis 工具类
 *
 * @author ruoyi
 **/
@Component
public class LocalCache {
    private final CacheMap<String, LoginUser> cacheMap = new CacheMap<>();

    /**
     * 缓存基本的对象，Integer、String、实体类等
     *
     * @param key 缓存的键值
     * @param value 缓存的值
     * @param timeout 时间
     */
    public void setCacheObject(String key, LoginUser value, long timeout) {
        cacheMap.set(key, value, timeout);
    }

    /**
     * 获得缓存的基本对象。
     *
     * @param key 缓存键值
     * @return 缓存键值对应的数据
     */
    public LoginUser getCacheObject(final String key) {
        return cacheMap.get(key);
    }

    /**
     * 删除单个对象
     */
    public void deleteObject(final String key) {
        cacheMap.remove(key);
    }

}
