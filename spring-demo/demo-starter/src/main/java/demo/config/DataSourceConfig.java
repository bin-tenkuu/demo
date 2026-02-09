package demo.config;

import com.baomidou.mybatisplus.autoconfigure.MybatisPlusProperties;
import demo.autoconfigure.MybatisUtil;
import lombok.RequiredArgsConstructor;
import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.SqlSessionTemplate;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.sql.DataSource;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/12/02
 */
@Configuration
@RequiredArgsConstructor
@EnableConfigurationProperties({MybatisPlusProperties.class})
@MapperScan(basePackages = "demo.mapper", sqlSessionTemplateRef = "sqlSessionTemplate")
public class DataSourceConfig {
    private final MybatisUtil mybatisUtil;

    @Bean(name = "sqlSessionFactory")
    public SqlSessionFactory emsSqlSessionFactory(DataSource dataSource) throws Exception {
        var sessionFactory = mybatisUtil.sqlSessionFactoryBean(dataSource);
        sessionFactory.setPlugins(mybatisUtil.interceptor(
        ));
        mybatisUtil.modify(sessionFactory, new String[]{"classpath*:/mapper/*.xml"});
        return sessionFactory.getObject();
    }

    @Bean(name = "sqlSessionTemplate")
    public SqlSessionTemplate emsSqlSessionTemplate(
            @Qualifier("sqlSessionFactory") SqlSessionFactory sqlSessionFactory
    ) {
        return new SqlSessionTemplate(sqlSessionFactory);
    }

}
