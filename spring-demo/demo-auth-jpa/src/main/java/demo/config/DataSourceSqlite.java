package demo.config;

import jakarta.persistence.EntityManagerFactory;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.orm.jpa.EntityManagerFactoryBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.orm.jpa.JpaTransactionManager;
import org.springframework.orm.jpa.LocalContainerEntityManagerFactoryBean;
import org.springframework.stereotype.Component;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;
import org.sqlite.SQLiteDataSource;

import javax.sql.DataSource;

/**
 * @author bin
 * @since 2025/12/15
 */
@Component
@RequiredArgsConstructor
@EnableTransactionManagement
@EnableJpaRepositories(
        entityManagerFactoryRef = "sqliteEntityManagerFactory",
        transactionManagerRef = "sqliteTransactionManager",
        basePackages = "demo.repository"
)
public class DataSourceSqlite {

    @Bean("sqliteDataSource")
    @ConfigurationProperties("spring.datasource.sqlite")
    public SQLiteDataSource dataSource() {
        return new SQLiteDataSource();
    }

    @Bean(name = "sqliteEntityManagerFactory")
    public LocalContainerEntityManagerFactoryBean managerFactory(
            EntityManagerFactoryBuilder builder,
            @Qualifier("sqliteDataSource") DataSource dataSource
    ) {
        return builder
                .dataSource(dataSource)
                .packages("demo.entity")
                .persistenceUnit("sqlite")
                .build();
    }

    @Bean(name = "sqliteTransactionManager")
    public PlatformTransactionManager transactionManager(
            @Qualifier("sqliteEntityManagerFactory") EntityManagerFactory customerEntityManagerFactory
    ) {
        return new JpaTransactionManager(customerEntityManagerFactory);
    }

}
