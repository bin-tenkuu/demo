package demo.starter.controller;

import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.cache.CacheManager;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.cache.concurrent.ConcurrentMapCache;
import org.springframework.cache.support.SimpleCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * @author bin
 * @since 2026/02/09
 */
@Tag(name = "cache")
@RestController
@RequestMapping("/v1/cache")
@EnableCaching
public class CacheController {
    @Bean
    public CacheManager cacheManager() {
        SimpleCacheManager cacheManager = new SimpleCacheManager();
        cacheManager.setCaches(List.of(
                new ConcurrentMapCache("cache")
        ));
        return cacheManager;
    }

    @Cacheable(cacheNames = "cache")
    @GetMapping("/get")
    public String cacheAble(String key) throws InterruptedException {
        Thread.sleep(3000);
        return "cacheAble:" + key;
    }

    @CacheEvict(cacheNames = "cache", allEntries = true, beforeInvocation = true)
    public String cacheEvict(String key) throws InterruptedException {
        Thread.sleep(3000);
        return "cacheEvict:" + key;
    }
}
