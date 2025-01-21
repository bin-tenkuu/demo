package demo.autoconfigure;

import com.baomidou.mybatisplus.autoconfigure.MybatisPlusProperties;
import com.baomidou.mybatisplus.autoconfigure.SpringBootVFS;
import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.extension.plugins.MybatisPlusInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.InnerInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.PaginationInnerInterceptor;
import com.baomidou.mybatisplus.extension.spring.MybatisSqlSessionFactoryBean;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.apache.ibatis.session.ExecutorType;
import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.SqlSessionTemplate;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

import javax.sql.DataSource;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/12/10
 */
@RequiredArgsConstructor
@EnableConfigurationProperties({MybatisPlusProperties.class})
public class MybatisUtil {
    private final MybatisPlusProperties properties;

    // region 1初始化
    public void modify(MybatisSqlSessionFactoryBean factory, String[] mapperLocations) {
        factory.setGlobalConfig(properties.getGlobalConfig());
        val configuration = new MybatisConfiguration();
        val coreConfiguration = properties.getConfiguration();
        properties.setMapperLocations(mapperLocations);
        factory.setMapperLocations(properties.resolveMapperLocations());
        if (coreConfiguration != null) {
            coreConfiguration.applyTo(configuration);
        }
        factory.setConfiguration(configuration);
    }

    public MybatisPlusInterceptor interceptor(InnerInterceptor... interceptors) {
        val interceptor = new MybatisPlusInterceptor();
        for (InnerInterceptor innerInterceptor : interceptors) {
            interceptor.addInnerInterceptor(innerInterceptor);
        }
        interceptor.addInnerInterceptor(new PaginationInnerInterceptor());

        return interceptor;
    }
    // endregion

    public MybatisSqlSessionFactoryBean sqlSessionFactoryBean(DataSource dataSource) {
        val factory = new MybatisSqlSessionFactoryBean();
        factory.setDataSource(dataSource);
        factory.setVfs(SpringBootVFS.class);
        modify(factory, new String[]{"classpath*:mapper/*.xml"});
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
