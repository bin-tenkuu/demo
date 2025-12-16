package demo.config;

import demo.autoconfigure.MybatisUtil;
import lombok.RequiredArgsConstructor;
import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.SqlSessionTemplate;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;
import org.sqlite.SQLiteDataSource;

import javax.sql.DataSource;

/**
 * @author bin
 * @since 2025/12/15
 */
@Component
@RequiredArgsConstructor
@MapperScan(basePackages = "demo.mapper", sqlSessionTemplateRef = "sqliteSqlSessionTemplate")
public class DataSourceSqlite {
    private final MybatisUtil mybatisUtil;

    @Bean("sqliteDataSource")
    @ConfigurationProperties("spring.datasource.sqlite")
    public SQLiteDataSource dataSource() {
        return new SQLiteDataSource();
    }

    @Bean(name = "sqliteSqlSessionFactory")
    public SqlSessionFactory sqlSessionFactory(
            @Qualifier("sqliteDataSource") DataSource dataSource
    ) throws Exception {
        var factoryBean = mybatisUtil.sqlSessionFactoryBean(dataSource);
        factoryBean.setPlugins(mybatisUtil.interceptor());
        mybatisUtil.modify(factoryBean, new String[]{"classpath*:/mapper/*.xml"});
        return factoryBean.getObject();
    }

    @Bean(name = "sqliteSqlSessionTemplate")
    public SqlSessionTemplate sqlSessionTemplate(
            @Qualifier("sqliteSqlSessionFactory") SqlSessionFactory sqlSessionFactory
    ) {
        return mybatisUtil.sqlSessionTemplate(sqlSessionFactory);
    }

}
