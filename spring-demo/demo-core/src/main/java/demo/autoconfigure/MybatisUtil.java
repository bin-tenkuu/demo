package demo.autoconfigure;

import com.baomidou.mybatisplus.autoconfigure.MybatisPlusProperties;
import com.baomidou.mybatisplus.autoconfigure.SpringBootVFS;
import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.handlers.MetaObjectHandler;
import com.baomidou.mybatisplus.core.injector.AbstractSqlInjector;
import com.baomidou.mybatisplus.extension.plugins.MybatisPlusInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.InnerInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.PaginationInnerInterceptor;
import com.baomidou.mybatisplus.extension.spring.MybatisSqlSessionFactoryBean;
import demo.autoconfigure.mybatis.GeneralSqlInjector;
import demo.autoconfigure.mybatis.MetaObjectHandlers;
import lombok.RequiredArgsConstructor;
import org.apache.ibatis.session.ExecutorType;
import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.SqlSessionTemplate;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

import javax.sql.DataSource;
import java.util.List;

/// @author bin
/// @version 1.0.0
/// @since 2024/12/10
@RequiredArgsConstructor
@EnableConfigurationProperties({MybatisPlusProperties.class})
public class MybatisUtil {
    private final MybatisPlusProperties properties;
    private final List<AbstractSqlInjector> sqlInjectors;
    private final List<MetaObjectHandler> metaObjectHandlers;

    public void modify(MybatisSqlSessionFactoryBean factory, String[] mapperLocations) {
        var globalConfig = properties.getGlobalConfig();
        factory.setGlobalConfig(globalConfig);
        globalConfig.setSqlInjector(new GeneralSqlInjector(sqlInjectors));
        globalConfig.setMetaObjectHandler(new MetaObjectHandlers(metaObjectHandlers));
        var configuration = new MybatisConfiguration();
        var coreConfiguration = properties.getConfiguration();
        properties.setMapperLocations(mapperLocations);
        factory.setMapperLocations(properties.resolveMapperLocations());
        if (coreConfiguration != null) {
            coreConfiguration.applyTo(configuration);
        }
        factory.setConfiguration(configuration);
    }

    public MybatisPlusInterceptor interceptor(InnerInterceptor... interceptors) {
        var interceptor = new MybatisPlusInterceptor();
        for (InnerInterceptor innerInterceptor : interceptors) {
            interceptor.addInnerInterceptor(innerInterceptor);
        }
        interceptor.addInnerInterceptor(new PaginationInnerInterceptor());

        return interceptor;
    }

    public MybatisSqlSessionFactoryBean sqlSessionFactoryBean(DataSource dataSource) {
        var factory = new MybatisSqlSessionFactoryBean();
        factory.setDataSource(dataSource);
        factory.setVfs(SpringBootVFS.class);
        return factory;
    }

    public SqlSessionTemplate sqlSessionTemplate(SqlSessionFactory sqlSessionFactory) {
        ExecutorType executorType = properties.getExecutorType();
        if (executorType != null) {
            return new SqlSessionTemplate(sqlSessionFactory, executorType);
        } else {
            return new SqlSessionTemplate(sqlSessionFactory);
        }
    }

}
